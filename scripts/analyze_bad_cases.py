#!/usr/bin/env python
"""
Bad Case Dynamic Analysis for Customer Service Agent (Production Version).

This script performs full-scale model rollouts in the CustomerServiceEnv
to identify all failure patterns across the entire training dataset.

Features:
  - Supports full dataset evaluation (4000+ playbooks)
  - No hard limits on dialogue length - captures errors in deep conversations
  - Progress tracking with ETA estimation
  - Multi-dimensional statistics (by scenario, by dialogue length)
  - Detailed reasoning trace capture for debugging

Usage:
    # Full dataset evaluation (default)
    python scripts/analyze_bad_cases.py \
        --playbook_path outputs/playbooks_all.json \
        --checkpoint_path outputs/hf_checkpoints/epoch_40

    # Sample evaluation (for quick testing)
    python scripts/analyze_bad_cases.py \
        --playbook_path outputs/playbooks_all.json \
        --checkpoint_path outputs/hf_checkpoints/epoch_40 \
        --num_samples 100

    # Short dialogues only (legacy mode)
    python scripts/analyze_bad_cases.py \
        --playbook_path outputs/playbooks_all.json \
        --checkpoint_path outputs/hf_checkpoints/epoch_40 \
        --max_rl_steps 5
"""

import os
import re
import json
import random
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict
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


# =============================================================================
# Constants
# =============================================================================

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

# Dialogue length buckets for statistics
DIALOGUE_LENGTH_BUCKETS = [
    (1, 5, "1-5 steps"),
    (6, 10, "6-10 steps"),
    (11, 15, "11-15 steps"),
    (16, 20, "16-20 steps"),
    (21, 999, "21+ steps"),
]


# =============================================================================
# Data Structures
# =============================================================================

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
    total_failures: int = 0
    success_rate: float = 0.0
    bad_cases: List[RolloutTrace] = field(default_factory=list)
    top_error_actions: List[Tuple[str, int]] = field(default_factory=list)
    scenario_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)
    length_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)


# =============================================================================
# Model Output Parsing
# =============================================================================

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
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('action:') or line.startswith('<action>'):
                action = line.replace('action:', '').replace('<action>', '').replace('</action>', '').strip()
                break

    return action, reasoning


# =============================================================================
# Prompt Building
# =============================================================================

def build_prompt_from_observation(
    observation: Dict[str, Any],
    dialogue_history: List[Dict[str, str]],
    action_history: Optional[List[str]] = None,
    history_length: int = 5
) -> str:
    """
    Build the prompt for model inference from environment observation.

    Uses CustomerServicePromptBuilder from rl_interfaces.py to ensure
    exact match with training-time prompts.
    """
    return CustomerServicePromptBuilder.build(
        observation=observation,
        action_history=action_history,
        history_length=history_length
    )


# =============================================================================
# Model Loading
# =============================================================================

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


# =============================================================================
# Response Generation
# =============================================================================

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


# =============================================================================
# Rollout Execution
# =============================================================================

def run_rollout(
    env,
    playbook_id: str,
    model,
    tokenizer,
    temperature: float = 0.4,
    max_tokens: int = 512,
    max_steps: int = 30,
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


# =============================================================================
# Statistics Helpers
# =============================================================================

def get_length_bucket(steps: int) -> str:
    """Get the dialogue length bucket for a given step count."""
    for min_steps, max_steps, label in DIALOGUE_LENGTH_BUCKETS:
        if min_steps <= steps <= max_steps:
            return label
    return "21+ steps"


def init_length_breakdown() -> Dict[str, Dict[str, int]]:
    """Initialize the length breakdown dictionary."""
    return {
        label: {'total': 0, 'won': 0, 'failed': 0, 'errors': 0}
        for _, _, label in DIALOGUE_LENGTH_BUCKETS
    }


# =============================================================================
# Progress Reporting
# =============================================================================

class ProgressReporter:
    """Progress reporter with ETA estimation."""

    def __init__(self, total: int, report_interval: int = 50):
        self.total = total
        self.report_interval = report_interval
        self.start_time = time.time()
        self.processed = 0
        self.last_report_time = self.start_time

    def update(self, count: int = 1):
        """Update progress counter."""
        self.processed += count

    def should_report(self) -> bool:
        """Check if we should report progress."""
        return self.processed % self.report_interval == 0 or self.processed == self.total

    def get_progress_line(self) -> str:
        """Get formatted progress line with ETA."""
        elapsed = time.time() - self.start_time
        rate = self.processed / elapsed if elapsed > 0 else 0

        if rate > 0:
            remaining = (self.total - self.processed) / rate
            eta_str = self._format_time(remaining)
        else:
            eta_str = "N/A"

        pct = self.processed / self.total * 100 if self.total > 0 else 0

        return f"Progress: {self.processed}/{self.total} ({pct:.1f}%) | ETA: {eta_str} | Rate: {rate:.2f} samples/s"

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_bad_cases(
    playbook_path: str,
    checkpoint_path: str,
    num_samples: int = 0,
    output_path: str = "outputs/all_bad_cases_trace.json",
    max_rl_steps: int = 999,
    seed: int = 42,
    temperature: float = 0.4,
    report_interval: int = 50
) -> BadCaseReport:
    """
    Main analysis function for full-scale evaluation.

    Args:
        playbook_path: Path to playbooks JSON file
        checkpoint_path: Path to HF checkpoint directory
        num_samples: Number of samples to analyze (0 = all)
        output_path: Output path for trace JSON
        max_rl_steps: Maximum RL steps filter (999 = no filter)
        seed: Random seed for sampling
        temperature: Sampling temperature (should match training)
        report_interval: Progress report interval

    Returns:
        BadCaseReport with analysis results
    """
    from etl.customer_service_env import CustomerServiceEnv

    logger.info("=" * 70)
    logger.info("Customer Service Agent - Bad Case Analysis (Production Mode)")
    logger.info("=" * 70)
    logger.info(f"Playbook: {playbook_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Max RL steps filter: {max_rl_steps if max_rl_steps < 999 else 'No limit'}")
    logger.info(f"Num samples: {num_samples if num_samples > 0 else 'All'}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 70)

    # Load playbooks
    with open(playbook_path, 'r', encoding='utf-8') as f:
        all_playbooks = json.load(f)
    logger.info(f"Loaded {len(all_playbooks)} total playbooks")

    # Filter playbooks (only filter out 'unknown' scenario, allow all dialogue lengths)
    filtered_playbooks = [
        p for p in all_playbooks
        if p.get('rl_steps', 0) <= max_rl_steps
        and p.get('scenario', 'unknown') != 'unknown'
    ]
    logger.info(f"After filtering: {len(filtered_playbooks)} playbooks")

    # Sample or use all
    random.seed(seed)
    if num_samples > 0 and len(filtered_playbooks) > num_samples:
        sample_playbooks = random.sample(filtered_playbooks, num_samples)
    else:
        sample_playbooks = filtered_playbooks

    total_to_process = len(sample_playbooks)
    logger.info(f"Will process {total_to_process} playbooks")

    # Initialize environment and model
    env = CustomerServiceEnv(playbook_path)
    model, tokenizer = load_model(checkpoint_path)

    # Initialize report
    report = BadCaseReport(timestamp=datetime.now().isoformat())
    report.total_episodes = total_to_process

    # Statistics counters
    action_error_counter = Counter()
    scenario_stats = defaultdict(lambda: {'total': 0, 'won': 0, 'failed': 0, 'errors': 0})
    length_stats = init_length_breakdown()

    # Progress reporter
    progress = ProgressReporter(total_to_process, report_interval)

    # Run rollouts
    start_time = time.time()

    for i, playbook in enumerate(sample_playbooks):
        playbook_id = playbook['playbook_id']

        try:
            trace = run_rollout(
                env=env,
                playbook_id=playbook_id,
                model=model,
                tokenizer=tokenizer,
                temperature=temperature,
                max_steps=30
            )

            # Update scenario statistics
            scenario = trace.scenario
            scenario_stats[scenario]['total'] += 1
            if trace.won:
                scenario_stats[scenario]['won'] += 1
            else:
                scenario_stats[scenario]['failed'] += 1
                report.total_failures += 1

                # Collect error actions
                for err in trace.error_actions:
                    action_error_counter[err['action']] += 1
                    scenario_stats[scenario]['errors'] += 1

                # Add to bad cases
                report.bad_cases.append(trace)

            # Update length statistics
            length_bucket = get_length_bucket(trace.rl_steps)
            length_stats[length_bucket]['total'] += 1
            if trace.won:
                length_stats[length_bucket]['won'] += 1
            else:
                length_stats[length_bucket]['failed'] += 1
                length_stats[length_bucket]['errors'] += len(trace.error_actions)

        except Exception as e:
            logger.error(f"Error processing {playbook_id}: {e}")
            continue

        # Update progress
        progress.update()

        if progress.should_report():
            logger.info(progress.get_progress_line())

    # Calculate success rate
    total_time = time.time() - start_time
    if report.total_episodes > 0:
        report.success_rate = (report.total_episodes - report.total_failures) / report.total_episodes * 100

    # Top error actions
    report.top_error_actions = action_error_counter.most_common(10)
    report.scenario_breakdown = dict(scenario_stats)
    report.length_breakdown = length_stats

    # Prepare output data
    trace_data = {
        'timestamp': report.timestamp,
        'config': {
            'playbook_path': playbook_path,
            'checkpoint_path': checkpoint_path,
            'num_samples': num_samples,
            'max_rl_steps': max_rl_steps,
            'temperature': temperature,
            'seed': seed
        },
        'summary': {
            'total_episodes': report.total_episodes,
            'total_failures': report.total_failures,
            'success_rate': f"{report.success_rate:.2f}%",
            'total_time_seconds': round(total_time, 2),
            'avg_time_per_episode': round(total_time / report.total_episodes, 3) if report.total_episodes > 0 else 0,
            'top_error_actions': report.top_error_actions,
        },
        'scenario_breakdown': {
            scenario: {
                'total': stats['total'],
                'won': stats['won'],
                'failed': stats['failed'],
                'success_rate': f"{stats['won'] / stats['total'] * 100:.1f}%" if stats['total'] > 0 else "N/A",
                'total_errors': stats['errors']
            }
            for scenario, stats in scenario_stats.items()
        },
        'length_breakdown': {
            bucket: {
                'total': stats['total'],
                'won': stats['won'],
                'failed': stats['failed'],
                'success_rate': f"{stats['won'] / stats['total'] * 100:.1f}%" if stats['total'] > 0 else "N/A",
                'total_errors': stats['errors']
            }
            for bucket, stats in length_stats.items()
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

    # Save trace
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trace_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Trace saved to {output_path}")

    # Print summary
    print_summary(report, total_time)

    return report


def print_summary(report: BadCaseReport, total_time: float):
    """Print detailed analysis summary."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total episodes analyzed: {report.total_episodes}")
    logger.info(f"Total failures: {report.total_failures}")
    logger.info(f"Overall success rate: {report.success_rate:.2f}%")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    logger.info("")

    # Scenario breakdown
    logger.info("-" * 70)
    logger.info("SCENARIO BREAKDOWN")
    logger.info("-" * 70)
    logger.info(f"{'Scenario':<15} {'Total':>8} {'Won':>8} {'Failed':>8} {'Success%':>10} {'Errors':>8}")
    logger.info("-" * 70)
    for scenario, stats in sorted(report.scenario_breakdown.items()):
        success_pct = stats['won'] / stats['total'] * 100 if stats['total'] > 0 else 0
        logger.info(f"{scenario:<15} {stats['total']:>8} {stats['won']:>8} {stats['failed']:>8} {success_pct:>9.1f}% {stats['errors']:>8}")
    logger.info("")

    # Length breakdown
    logger.info("-" * 70)
    logger.info("DIALOGUE LENGTH BREAKDOWN (by rl_steps)")
    logger.info("-" * 70)
    logger.info(f"{'Length':<15} {'Total':>8} {'Won':>8} {'Failed':>8} {'Success%':>10} {'Errors':>8}")
    logger.info("-" * 70)
    for bucket, stats in report.length_breakdown.items():
        if stats['total'] > 0:
            success_pct = stats['won'] / stats['total'] * 100
            logger.info(f"{bucket:<15} {stats['total']:>8} {stats['won']:>8} {stats['failed']:>8} {success_pct:>9.1f}% {stats['errors']:>8}")
    logger.info("")

    # Top error actions
    logger.info("-" * 70)
    logger.info("TOP 10 ERROR ACTIONS")
    logger.info("-" * 70)
    for action, count in report.top_error_actions:
        logger.info(f"  {action}: {count} errors")
    logger.info("")

    logger.info("=" * 70)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze bad cases in Customer Service Agent (Production Mode)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full dataset evaluation
  python scripts/analyze_bad_cases.py --checkpoint_path outputs/hf_checkpoints/epoch_40

  # Sample 100 playbooks for quick testing
  python scripts/analyze_bad_cases.py --num_samples 100 --checkpoint_path outputs/hf_checkpoints/epoch_40

  # Short dialogues only (legacy mode)
  python scripts/analyze_bad_cases.py --max_rl_steps 5 --checkpoint_path outputs/hf_checkpoints/epoch_40
        """
    )
    parser.add_argument('--playbook_path', type=str, default='outputs/playbooks_all.json',
                        help='Path to playbooks JSON file')
    parser.add_argument('--checkpoint_path', type=str, default='outputs/hf_checkpoints/epoch_40',
                        help='Path to HF checkpoint directory')
    parser.add_argument('--num_samples', type=int, default=0,
                        help='Number of samples to analyze (0 = all playbooks)')
    parser.add_argument('--max_rl_steps', type=int, default=999,
                        help='Maximum RL steps filter (999 = no filter, use 5 for short-only mode)')
    parser.add_argument('--output_path', type=str, default='outputs/all_bad_cases_trace.json',
                        help='Output path for trace JSON')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--temperature', type=float, default=0.4,
                        help='Sampling temperature (should match training)')
    parser.add_argument('--report_interval', type=int, default=50,
                        help='Progress report interval')

    args = parser.parse_args()

    analyze_bad_cases(
        playbook_path=args.playbook_path,
        checkpoint_path=args.checkpoint_path,
        num_samples=args.num_samples,
        output_path=args.output_path,
        max_rl_steps=args.max_rl_steps,
        seed=args.seed,
        temperature=args.temperature,
        report_interval=args.report_interval
    )


if __name__ == '__main__':
    main()