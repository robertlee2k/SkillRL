#!/usr/bin/env python3
"""
Full evaluation of a trained model against ALL playbooks.

Runs each playbook through the CustomerServiceEnv with vLLM inference,
using the exact same prompt construction and action projection as training.

Usage:
    python scripts/full_eval.py \
        --model_path ~/data/SkillRL/skillrl_models/customer_service/step_210 \
        --playbook_path outputs/playbooks_all_fixed_v2.json \
        --output_path outputs/eval_step_210.json \
        --max_steps 20 \
        --tensor_parallel_size 4 \
        --temperature 0.4
"""

import os
import sys
import json
import time
import argparse
import logging
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.customer_service_env import CustomerServiceEnv, VALID_SKILLS
from etl.rl_interfaces import (
    CustomerServicePromptBuilder,
    customer_service_projection,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_messages_for_step(env: CustomerServiceEnv) -> List[Dict[str, str]]:
    """Build chat messages from current env state, matching training-time format."""
    obs = env._get_observation()
    obs["scenario"] = env.state.scenario
    obs["action_history"] = env.state.action_history
    obs["dialogue_history"] = env.state.dialogue_history

    available_skills = obs.get("available_skills", list(VALID_SKILLS))
    system_content = CustomerServicePromptBuilder._build_system_prompt_content(available_skills)
    user_content = CustomerServicePromptBuilder._build_user_prompt_content(
        scenario=obs["scenario"],
        slots=obs.get("slots", {}),
        buyer_text=obs.get("buyer_text", ""),
        dialogue_history=obs.get("dialogue_history"),
        action_history=obs.get("action_history"),
        history_length=5,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def run_single_episode(
    env: CustomerServiceEnv,
    llm,
    tokenizer,
    sampling_params,
    playbook_id: str,
    max_steps: int,
) -> Dict[str, Any]:
    """Run a single episode and return detailed results."""
    obs, info = env.reset(playbook_id=playbook_id)
    scenario = env.state.scenario
    target_rl_steps = env.current_playbook.get("rl_steps", 0)

    action_log = []
    step_rewards = []
    total_step_reward = 0.0

    for step_idx in range(max_steps):
        if env.state.done:
            break

        # Build prompt (same as training)
        messages = build_messages_for_step(env)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate
        from vllm import SamplingParams as _SP  # avoid top-level import issues

        outputs = llm.generate([prompt_text], sampling_params)
        raw_output = outputs[0].outputs[0].text

        # Parse action (same projection as training, no fallback)
        available_skills = obs.get("available_skills", [])
        actions, valids = customer_service_projection(
            [raw_output], [available_skills]
        )
        action = actions[0]
        valid = valids[0]

        # If projection returned INVALID_ACTION, still pass it to env
        # (env will handle it via Tier 1 / Tier 2 logic)
        if action == "INVALID_ACTION":
            action_for_env = "__INVALID__"  # will trigger Tier 1
        else:
            action_for_env = action

        # Step environment
        obs, step_reward, done, step_info = env.step(action_for_env)
        total_step_reward += step_reward

        action_log.append({
            "step": step_idx,
            "action": action,
            "valid": valid,
            "step_reward": round(step_reward, 3),
            "raw_output_preview": raw_output[:200],
        })
        step_rewards.append(step_reward)

    # Episode reward
    episode_reward = env.compute_episode_reward()

    # Classify failure reason
    failure_reason = None
    if not env.state.won:
        invalid_count = sum(1 for a in action_log if a["valid"] == 0)
        if env.state.patience <= 0:
            failure_reason = "patience_exhausted"
        elif step_idx >= max_steps - 1 and not env.state.done:
            failure_reason = "max_steps_reached"
        elif invalid_count > 0:
            failure_reason = f"invalid_actions({invalid_count})"
        else:
            failure_reason = "env_not_done"

    return {
        "playbook_id": playbook_id,
        "scenario": scenario,
        "target_rl_steps": target_rl_steps,
        "actual_steps": len(action_log),
        "won": env.state.won,
        "done": env.state.done,
        "patience_remaining": env.state.patience,
        "episode_reward": round(episode_reward, 3),
        "total_step_reward": round(total_step_reward, 3),
        "failure_reason": failure_reason,
        "action_sequence": [a["action"] for a in action_log],
        "valid_action_rate": (
            sum(1 for a in action_log if a["valid"] == 1) / len(action_log)
            if action_log else 0.0
        ),
        "action_log": action_log,
    }


def main():
    parser = argparse.ArgumentParser(description="Full model evaluation on all playbooks")
    parser.add_argument("--model_path", required=True, help="Path to merged HF model")
    parser.add_argument("--playbook_path", required=True, help="Path to playbooks JSON")
    parser.add_argument("--output_path", default="outputs/eval_results.json", help="Output path")
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of playbooks (for testing)")
    args = parser.parse_args()

    # ========== Load model ==========
    logger.info(f"Loading model from {args.model_path}")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=8192,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=0.95,
    )

    # ========== Load playbooks ==========
    logger.info(f"Loading playbooks from {args.playbook_path}")
    with open(args.playbook_path, "r", encoding="utf-8") as f:
        playbooks = json.load(f)

    if args.limit:
        playbooks = playbooks[: args.limit]

    logger.info(f"Total playbooks to evaluate: {len(playbooks)}")

    # ========== Create env ==========
    env = CustomerServiceEnv(args.playbook_path)

    # ========== Run evaluation ==========
    results = []
    t0 = time.time()

    for idx, pb in enumerate(playbooks):
        pb_id = pb["playbook_id"]
        result = run_single_episode(
            env=env,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            playbook_id=pb_id,
            max_steps=args.max_steps,
        )
        results.append(result)

        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t0
            wins = sum(1 for r in results if r["won"])
            rate = wins / len(results) * 100
            eps = len(results) / elapsed
            logger.info(
                f"[{idx+1}/{len(playbooks)}] "
                f"success={rate:.1f}% ({wins}/{len(results)}) "
                f"speed={eps:.1f} ep/s "
                f"elapsed={elapsed:.0f}s"
            )

    elapsed_total = time.time() - t0

    # ========== Compute statistics ==========
    total = len(results)
    wins = sum(1 for r in results if r["won"])
    dones = sum(1 for r in results if r["done"])

    # By scenario
    scenario_stats = defaultdict(lambda: {"total": 0, "won": 0, "rewards": []})
    for r in results:
        s = r["scenario"]
        scenario_stats[s]["total"] += 1
        if r["won"]:
            scenario_stats[s]["won"] += 1
        scenario_stats[s]["rewards"].append(r["episode_reward"])

    scenario_summary = {}
    for s, st in scenario_stats.items():
        scenario_summary[s] = {
            "total": st["total"],
            "won": st["won"],
            "success_rate": round(st["won"] / st["total"] * 100, 1) if st["total"] > 0 else 0,
            "avg_episode_reward": round(sum(st["rewards"]) / len(st["rewards"]), 3),
        }

    # By target_rl_steps bucket
    bucket_stats = defaultdict(lambda: {"total": 0, "won": 0})
    for r in results:
        steps = r["target_rl_steps"]
        if steps <= 3:
            bucket = "1-3"
        elif steps <= 5:
            bucket = "4-5"
        elif steps <= 10:
            bucket = "6-10"
        elif steps <= 15:
            bucket = "11-15"
        else:
            bucket = "16-20"
        bucket_stats[bucket]["total"] += 1
        if r["won"]:
            bucket_stats[bucket]["won"] += 1

    bucket_summary = {}
    for b, st in sorted(bucket_stats.items()):
        bucket_summary[b] = {
            "total": st["total"],
            "won": st["won"],
            "success_rate": round(st["won"] / st["total"] * 100, 1) if st["total"] > 0 else 0,
        }

    # Failure reasons
    failure_reasons = Counter()
    for r in results:
        if not r["won"] and r["failure_reason"]:
            failure_reasons[r["failure_reason"]] += 1

    # Most common failed actions
    failed_action_counter = Counter()
    for r in results:
        if not r["won"]:
            for a in r["action_log"]:
                if a["valid"] == 0:
                    failed_action_counter[a["action"]] += 1

    # Valid action rate distribution
    valid_rates = [r["valid_action_rate"] for r in results]
    avg_valid_rate = sum(valid_rates) / len(valid_rates) if valid_rates else 0

    # ========== Build report ==========
    report = {
        "config": {
            "model_path": args.model_path,
            "playbook_path": args.playbook_path,
            "max_steps": args.max_steps,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
        "summary": {
            "total_playbooks": total,
            "total_won": wins,
            "total_done": dones,
            "success_rate": round(wins / total * 100, 1),
            "completion_rate": round(dones / total * 100, 1),
            "avg_episode_reward": round(
                sum(r["episode_reward"] for r in results) / total, 3
            ),
            "avg_valid_action_rate": round(avg_valid_rate * 100, 1),
            "elapsed_seconds": round(elapsed_total, 1),
            "speed_episodes_per_sec": round(total / elapsed_total, 2),
        },
        "by_scenario": scenario_summary,
        "by_target_steps": bucket_summary,
        "failure_reasons": dict(failure_reasons.most_common(20)),
        "top_invalid_actions": dict(failed_action_counter.most_common(20)),
        # Store individual results without verbose action_log for file size
        "results": [
            {k: v for k, v in r.items() if k != "action_log"}
            for r in results
        ],
    }

    # Save bad cases with full detail
    bad_cases = [r for r in results if not r["won"]]
    report["bad_case_count"] = len(bad_cases)

    # ========== Print summary ==========
    print("\n" + "=" * 70)
    print("FULL EVALUATION REPORT")
    print("=" * 70)
    print(f"Model:          {args.model_path}")
    print(f"Playbooks:      {total}")
    print(f"Success Rate:   {report['summary']['success_rate']}% ({wins}/{total})")
    print(f"Completion Rate:{report['summary']['completion_rate']}% ({dones}/{total})")
    print(f"Avg Reward:     {report['summary']['avg_episode_reward']}")
    print(f"Valid Action %: {report['summary']['avg_valid_action_rate']}%")
    print(f"Time:           {elapsed_total:.0f}s ({total/elapsed_total:.1f} ep/s)")

    print(f"\n--- By Scenario ---")
    for s, st in sorted(scenario_summary.items()):
        print(f"  {s:12s}: {st['success_rate']:5.1f}% ({st['won']}/{st['total']}), avg_reward={st['avg_episode_reward']}")

    print(f"\n--- By Target Steps ---")
    for b, st in sorted(bucket_summary.items()):
        print(f"  {b:8s}: {st['success_rate']:5.1f}% ({st['won']}/{st['total']})")

    print(f"\n--- Top Failure Reasons ---")
    for reason, count in failure_reasons.most_common(10):
        print(f"  {reason:30s}: {count}")

    print(f"\n--- Top Invalid Actions ---")
    for action, count in failed_action_counter.most_common(10):
        print(f"  {action:30s}: {count}")

    print("=" * 70)

    # ========== Save ==========
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Report saved to {args.output_path}")

    # Save bad cases separately (with full action_log)
    bad_cases_path = args.output_path.replace(".json", "_bad_cases.json")
    with open(bad_cases_path, "w", encoding="utf-8") as f:
        json.dump(bad_cases, f, ensure_ascii=False, indent=2)
    logger.info(f"Bad cases ({len(bad_cases)}) saved to {bad_cases_path}")


if __name__ == "__main__":
    main()
