#!/usr/bin/env python
"""
Bad Case Dynamic Analysis for Short Dialogues (1-5 RL steps).

This script performs real-time model rollouts in the CustomerServiceEnv
to identify failure patterns in short conversations.

Features:
  - Loads HF checkpoint using Transformers (works with concurrent training)
  - Filters short test playbooks (rl_steps <= 5, scenario != unknown)
  - Executes full rollouts with reasoning trace capture
  - Collects "death cause" analysis for failed short dialogues
  - Outputs detailed trace JSON and summary statistics

Usage:
    python scripts/analyze_bad_cases.py \
        --playbook_path outputs/playbooks_all.json \
        --checkpoint_path outputs/hf_checkpoints/epoch_40 \
        --num_samples 100 \
        --output_path outputs/short_bad_cases_trace.json
"""

import os
import re
import json
import random
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the correct prompt builder from training code
import sys
sys.path.insert(0, '/home/bo.li/SkillRL')
from etl.rl_interfaces import CustomerServicePromptBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# System prompt (must match training)
SYSTEM_PROMPT = """你是一名专业的电商客服智能助手，擅长处理售前咨询、物流查询和售后问题。
你的目标是高效解决买家问题，促成交易或妥善处理售后，同时避免激怒买家。

你需要在每一步选择正确的服务动作。你的输出格式必须是：

<tool_call>
[你的分析过程：买家想要什么？当前对话处于什么阶段？哪个动作最合适？]
<action>skill_id</action>

可用的动作 ID 包括：
- 通用: gen_greet, gen_empathize, gen_clarify, gen_verify_order, gen_hold, gen_transfer, gen_apologize, gen_close
- 售前: pre_query_product, pre_check_stock, pre_compare, pre_recommend, pre_answer_spec, pre_check_promo, pre_guide_purchase
- 物流: log_query_status, log_query_detail, log_estimate_arrival, log_modify_address, log_contact_courier, log_delay_notify, log_lost_claim
- 售后: aft_check_policy, aft_collect_evidence, aft_initiate_refund, aft_initiate_return, aft_initiate_exchange, aft_schedule_pickup, aft_track_progress, aft_compensate, aft_reject_explain
"""


@dataclass
class RolloutTrace:
    """Records a single rollout episode trace."""
    playbook_id: str
    scenario: str
    rl_steps: int
    won: bool
    total_steps: int
    dialogue_history: List[Dict[str, str]] = field(default_factory=list)
    action_history: List[str] = field(default_factory=list)
    reasoning_traces: List[str] = field(default_factory=list)
    error_actions: List[Dict[str, Any]] = field(default_factory=list)
    final_patience: int = 0


@dataclass
class BadCaseReport:
    """Aggregated bad case analysis report."""
    timestamp: str
    total_episodes: int = 0
    short_episodes: int = 0
    short_failures: int = 0
    short_success_rate: float = 0.0
    bad_cases: List[RolloutTrace] = field(default_factory=list)
    top_error_actions: List[Tuple[str, int]] = field(default_factory=list)
    scenario_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)


def parse_model_output(output: str) -> Tuple[str, Optional[str]]:
    """
    Parse model output to extract action and reasoning.

    Training uses: ◈...◈ for thinking and <action>...</action> for action.

    Returns:
        (action, reasoning) tuple. reasoning may be None.
    """
    # Extract action from <action>...</action>
    action_match = re.search(r'<action>(.*?)</action>', output, re.DOTALL)
    action = action_match.group(1).strip() if action_match else ""

    # Extract reasoning from ◈...◈ block (training format)
    reasoning_match = re.search(r'◈(.*?)◈', output, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None

    # Fallback: if no action tag, try to extract from text
    if not action:
        # Look for action-like patterns
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('action:') or line.startswith('<action>'):
                action = line.replace('action:', '').replace('<action>', '').replace('</action>', '').strip()
                break

    return action, reasoning


def build_prompt_from_observation(
    observation: Dict[str, Any],
    dialogue_history: List[Dict[str, str]],
    action_history: List[str] = None,
    history_length: int = 5
) -> str:
    """
    Build the prompt for model inference from environment observation.

    Uses CustomerServicePromptBuilder from rl_interfaces.py to ensure
    exact match with training-time prompts.

    Returns:
        Formatted prompt string
    """
    # Use the same prompt builder as training
    return CustomerServicePromptBuilder.build(
        observation=observation,
        action_history=action_history,
        history_length=history_length
    )


def load_model(checkpoint_path: str):
    """Load model using Transformers."""
    logger.info(f"Loading model from {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    logger.info(f"Model loaded on {model.device}")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    temperature: float = 0.4,
    max_new_tokens: int = 512
) -> str:
    """Generate response using the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response


def run_rollout(
    env,
    playbook_id: str,
    model,
    tokenizer,
    temperature: float = 0.4,
    max_tokens: int = 512,
    max_steps: int = 20,
    history_length: int = 5
) -> RolloutTrace:
    """
    Execute a single episode rollout with the model.

    Returns a RolloutTrace with full dialogue and reasoning history.
    """
    # Reset environment
    observation, info = env.reset(playbook_id=playbook_id)

    trace = RolloutTrace(
        playbook_id=playbook_id,
        scenario=info['scenario'],
        rl_steps=env.current_playbook.get('rl_steps', 0),
        won=False,
        total_steps=0
    )

    while not env.state.done and trace.total_steps < max_steps:
        # Build prompt using the same builder as training
        prompt = build_prompt_from_observation(
            observation=observation,
            dialogue_history=env.state.dialogue_history,
            action_history=env.state.action_history if env.state.action_history else None,
            history_length=history_length
        )

        # Generate response
        model_output = generate_response(
            model, tokenizer, prompt,
            temperature=temperature,
            max_new_tokens=max_tokens
        )

        # Parse action and reasoning
        action, reasoning = parse_model_output(model_output)

        # Record reasoning
        if reasoning:
            trace.reasoning_traces.append(reasoning)

        # Step environment
        prev_patience = env.state.patience
        obs, reward, done, step_info = env.step(action)

        # Record action
        trace.action_history.append(action)
        trace.total_steps += 1

        # Check if this was an error
        if step_info.get('fell_back', False):
            trace.error_actions.append({
                'step': trace.total_steps,
                'action': action,
                'reasoning': reasoning,
                'patience_before': prev_patience,
                'patience_after': step_info.get('patience', 0),
                'system_message': env.state.dialogue_history[-1].get('content', '') if env.state.dialogue_history else ''
            })

        observation = obs

    # Episode finished
    trace.won = env.state.won
    trace.dialogue_history = env.state.dialogue_history.copy()
    trace.final_patience = env.state.patience

    return trace


def analyze_bad_cases(
    playbook_path: str,
    checkpoint_path: str,
    num_samples: int = 100,
    output_path: str = "outputs/short_bad_cases_trace.json",
    max_rl_steps: int = 5,
    seed: int = 42,
    temperature: float = 0.4
) -> BadCaseReport:
    """
    Main analysis function.

    1. Filter short playbooks (rl_steps <= max_rl_steps, scenario != unknown)
    2. Run rollouts with the model
    3. Collect bad cases (failed short dialogues)
    4. Output trace and statistics
    """
    import sys
    sys.path.insert(0, '/home/bo.li/SkillRL')
    from etl.customer_service_env import CustomerServiceEnv

    logger.info("=" * 60)
    logger.info("Bad Case Analysis for Short Dialogues")
    logger.info("=" * 60)
    logger.info(f"Playbook: {playbook_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Max RL steps: {max_rl_steps}")
    logger.info(f"Num samples: {num_samples}")
    logger.info("=" * 60)

    # Load playbooks
    with open(playbook_path, 'r', encoding='utf-8') as f:
        all_playbooks = json.load(f)
    logger.info(f"Loaded {len(all_playbooks)} total playbooks")

    # Filter short playbooks
    short_playbooks = [
        p for p in all_playbooks
        if p.get('rl_steps', 0) <= max_rl_steps
        and p.get('scenario', 'unknown') != 'unknown'
    ]
    logger.info(f"Found {len(short_playbooks)} short playbooks (rl_steps <= {max_rl_steps})")

    # Sample
    random.seed(seed)
    if len(short_playbooks) > num_samples:
        sample_playbooks = random.sample(short_playbooks, num_samples)
    else:
        sample_playbooks = short_playbooks[:num_samples]

    logger.info(f"Sampled {len(sample_playbooks)} playbooks for analysis")

    # Initialize environment and model
    env = CustomerServiceEnv(playbook_path)
    model, tokenizer = load_model(checkpoint_path)

    # Run rollouts
    report = BadCaseReport(timestamp=datetime.now().isoformat())
    report.total_episodes = len(sample_playbooks)

    action_error_counter = Counter()
    scenario_stats = {}

    for i, playbook in enumerate(sample_playbooks):
        playbook_id = playbook['playbook_id']

        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{len(sample_playbooks)}")

        try:
            trace = run_rollout(
                env=env,
                playbook_id=playbook_id,
                model=model,
                tokenizer=tokenizer,
                temperature=temperature,
                max_steps=20
            )

            # Track statistics
            scenario = trace.scenario
            if scenario not in scenario_stats:
                scenario_stats[scenario] = {'total': 0, 'won': 0, 'failed': 0}
            scenario_stats[scenario]['total'] += 1

            report.short_episodes += 1

            if trace.won:
                scenario_stats[scenario]['won'] += 1
            else:
                scenario_stats[scenario]['failed'] += 1
                report.short_failures += 1

                # Collect error actions
                for err in trace.error_actions:
                    action_error_counter[err['action']] += 1

                # Add to bad cases
                report.bad_cases.append(trace)

        except Exception as e:
            logger.error(f"Error processing {playbook_id}: {e}")
            continue

    # Calculate success rate
    if report.short_episodes > 0:
        report.short_success_rate = (report.short_episodes - report.short_failures) / report.short_episodes * 100

    # Top error actions
    report.top_error_actions = action_error_counter.most_common(5)
    report.scenario_breakdown = scenario_stats

    # Save trace
    trace_data = {
        'timestamp': report.timestamp,
        'summary': {
            'total_episodes': report.total_episodes,
            'short_episodes': report.short_episodes,
            'short_failures': report.short_failures,
            'short_success_rate': f"{report.short_success_rate:.1f}%",
            'top_error_actions': report.top_error_actions,
            'scenario_breakdown': report.scenario_breakdown
        },
        'bad_cases': [
            {
                'playbook_id': bc.playbook_id,
                'scenario': bc.scenario,
                'rl_steps': bc.rl_steps,
                'total_steps': bc.total_steps,
                'won': bc.won,
                'final_patience': bc.final_patience,
                'action_history': bc.action_history,
                'error_actions': bc.error_actions,
                'reasoning_traces': bc.reasoning_traces[:3],  # First 3 reasoning traces
                'dialogue_history': bc.dialogue_history
            }
            for bc in report.bad_cases
        ]
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trace_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Trace saved to {output_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Short dialogues analyzed: {report.short_episodes}")
    logger.info(f"Short dialogue failures: {report.short_failures}")
    logger.info(f"Short dialogue success rate: {report.short_success_rate:.1f}%")
    logger.info("")
    logger.info("Scenario breakdown:")
    for scenario, stats in scenario_stats.items():
        success_rate = stats['won'] / stats['total'] * 100 if stats['total'] > 0 else 0
        logger.info(f"  {scenario}: {stats['won']}/{stats['total']} ({success_rate:.1f}% success)")
    logger.info("")
    logger.info("Top 5 error actions:")
    for action, count in report.top_error_actions:
        logger.info(f"  {action}: {count} errors")
    logger.info("=" * 60)

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze bad cases in short dialogues')
    parser.add_argument('--playbook_path', type=str, default='outputs/playbooks_all.json',
                        help='Path to playbooks JSON file')
    parser.add_argument('--checkpoint_path', type=str, default='outputs/hf_checkpoints/epoch_40',
                        help='Path to HF checkpoint directory')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to analyze')
    parser.add_argument('--max_rl_steps', type=int, default=5,
                        help='Maximum RL steps to consider as short dialogue')
    parser.add_argument('--output_path', type=str, default='outputs/short_bad_cases_trace.json',
                        help='Output path for trace JSON')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--temperature', type=float, default=0.4,
                        help='Sampling temperature (should match training)')

    args = parser.parse_args()

    analyze_bad_cases(
        playbook_path=args.playbook_path,
        checkpoint_path=args.checkpoint_path,
        num_samples=args.num_samples,
        output_path=args.output_path,
        max_rl_steps=args.max_rl_steps,
        seed=args.seed,
        temperature=args.temperature
    )


if __name__ == '__main__':
    main()