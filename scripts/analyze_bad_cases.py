#!/usr/bin/env python
"""
Bad Case Dynamic Analysis for Customer Service Agent (Multi-GPU Parallel Version).

This script performs full-scale model rollouts in the CustomerServiceEnv
to identify all failure patterns across the entire training dataset.

Features:
  - Multi-GPU parallel inference (8x speedup with 8 GPUs)
  - Memory-limited model loading (respects existing training jobs)
  - Supports full dataset evaluation (4000+ playbooks)
  - Progress tracking with ETA estimation
  - Multi-dimensional statistics (by scenario, by dialogue length)

Usage:
    # Full dataset evaluation with 8 GPUs (default)
    python scripts/analyze_bad_cases.py \
        --playbook_path outputs/playbooks_all.json \
        --checkpoint_path outputs/hf_checkpoints/epoch_40 \
        --num_gpus 8 \
        --max_memory_per_gpu 20

    # Single GPU mode (for testing)
    python scripts/analyze_bad_cases.py \
        --playbook_path outputs/playbooks_all.json \
        --checkpoint_path outputs/hf_checkpoints/epoch_40 \
        --num_gpus 1
"""

import os
import re
import json
import random
import time
import logging
import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from datetime import datetime
import traceback

# NOTE: ALL torch-related imports are inside worker functions to ensure
# CUDA_VISIBLE_DEVICES is set before any CUDA initialization in spawn mode.
# DO NOT add any imports here that transitively import torch, including:
#   - etl.rl_interfaces (imports agent_system.environments.base which imports torch)
#   - agent_system.* (imports torch)
#   - transformers (imports torch)

import sys
sys.path.insert(0, '/home/bo.li/SkillRL')
# CustomerServicePromptBuilder import moved inside GPUWorker._run_rollout()

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


def trace_to_dict(trace: RolloutTrace) -> Dict:
    """Convert RolloutTrace to serializable dict."""
    return {
        'playbook_id': trace.playbook_id,
        'scenario': trace.scenario,
        'rl_steps': trace.rl_steps,
        'total_steps': trace.total_steps,
        'won': trace.won,
        'final_patience': trace.final_patience,
        'action_history': trace.action_history,
        'error_actions': trace.error_actions,
        'reasoning_traces': trace.reasoning_traces[:3],
        'dialogue_history': trace.dialogue_history
    }


def dict_to_trace(d: Dict) -> RolloutTrace:
    """Convert dict back to RolloutTrace."""
    return RolloutTrace(
        playbook_id=d['playbook_id'],
        scenario=d['scenario'],
        rl_steps=d['rl_steps'],
        total_steps=d['total_steps'],
        won=d['won'],
        final_patience=d['final_patience'],
        action_history=d['action_history'],
        error_actions=d['error_actions'],
        reasoning_traces=d.get('reasoning_traces', []),
        dialogue_history=d['dialogue_history']
    )


# =============================================================================
# Model Output Parsing
# =============================================================================

def parse_model_output(output: str) -> Tuple[str, Optional[str]]:
    """Parse model output to extract action and reasoning."""
    action_match = re.search(r'<action>(.*?)</action>', output, re.DOTALL)
    action = action_match.group(1).strip() if action_match else ""

    reasoning_match = re.search(r'◈(.*?)◈', output, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None

    if not action:
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('action:') or line.startswith('<action>'):
                action = line.replace('action:', '').replace('<action>', '').replace('</action>', '').strip()
                break

    return action, reasoning


# =============================================================================
# GPU Worker Process
# =============================================================================

class GPUWorker:
    """Worker that processes rollouts on a specific GPU."""

    def __init__(
        self,
        gpu_id: int,
        checkpoint_path: str,
        playbook_path: str,
        max_memory_gb: int = 20,
        temperature: float = 0.4,
        max_steps: int = 30
    ):
        self.gpu_id = gpu_id
        self.checkpoint_path = checkpoint_path
        self.playbook_path = playbook_path
        self.max_memory_gb = max_memory_gb
        self.temperature = temperature
        self.max_steps = max_steps

        # Will be initialized in setup
        self.model = None
        self.tokenizer = None
        self.env = None
        self.torch = None  # Store torch reference for _run_rollout

    def setup(self):
        """Initialize model and environment on the assigned GPU."""
        # CUDA_VISIBLE_DEVICES already set in worker_process before torch import

        # Import torch and transformers here (after CUDA_VISIBLE_DEVICES is set)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from etl.customer_service_env import CustomerServiceEnv

        # Store torch reference for later use
        self.torch = torch

        # Load model with memory limit
        max_memory = {0: f"{self.max_memory_gb}GiB"}

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True
        )
        self.model.eval()

        # Load environment
        self.env = CustomerServiceEnv(self.playbook_path)

        print(f"[GPU {self.gpu_id}] Setup complete, model loaded with {self.max_memory_gb}GB limit")

    def process_batch(self, playbook_ids: List[str]) -> List[Dict]:
        """Process a batch of playbooks and return results."""
        results = []

        for playbook_id in playbook_ids:
            try:
                trace = self._run_rollout(playbook_id)
                results.append({
                    'success': True,
                    'trace': trace_to_dict(trace)
                })
            except Exception as e:
                results.append({
                    'success': False,
                    'playbook_id': playbook_id,
                    'error': str(e)
                })

        return results

    def _run_rollout(self, playbook_id: str) -> RolloutTrace:
        """Execute a single episode rollout."""
        # Import prompt builder here (after CUDA_VISIBLE_DEVICES is set in worker_process)
        from etl.rl_interfaces import CustomerServicePromptBuilder

        observation, info = self.env.reset(playbook_id=playbook_id)

        trace = RolloutTrace(
            playbook_id=playbook_id,
            scenario=info['scenario'],
            rl_steps=self.env.current_playbook.get('rl_steps', 0),
            won=False,
            total_steps=0
        )

        while not self.env.state.done and trace.total_steps < self.max_steps:
            prompt = CustomerServicePromptBuilder.build(
                observation=observation,
                action_history=self.env.state.action_history if self.env.state.action_history else None,
                history_length=5
            )

            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with self.torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=self.temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            model_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Parse action
            action, reasoning = parse_model_output(model_output)

            if reasoning:
                trace.reasoning_traces.append(reasoning)

            # Step environment
            prev_patience = self.env.state.patience
            obs, reward, done, step_info = self.env.step(action)

            trace.action_history.append(action)
            trace.total_steps += 1

            if step_info.get('fell_back', False):
                trace.error_actions.append({
                    'step': trace.total_steps,
                    'action': action,
                    'reasoning': reasoning,
                    'patience_before': prev_patience,
                    'patience_after': step_info.get('patience', 0),
                    'system_message': self.env.state.dialogue_history[-1].get('content', '') if self.env.state.dialogue_history else ''
                })

            observation = obs

        trace.won = self.env.state.won
        trace.dialogue_history = self.env.state.dialogue_history.copy()
        trace.final_patience = self.env.state.patience

        return trace


def worker_process(
    gpu_id: int,
    task_queue,
    result_queue,
    checkpoint_path: str,
    playbook_path: str,
    max_memory_gb: int,
    temperature: float,
    max_steps: int
):
    """Worker process entry point."""
    # CRITICAL: Set CUDA device BEFORE any torch/CUDA initialization
    # This must be the very first thing in the worker process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        worker = GPUWorker(
            gpu_id=gpu_id,
            checkpoint_path=checkpoint_path,
            playbook_path=playbook_path,
            max_memory_gb=max_memory_gb,
            temperature=temperature,
            max_steps=max_steps
        )
        worker.setup()

        while True:
            task = task_queue.get()

            if task is None:  # Poison pill
                result_queue.put(None)
                break

            batch_id, playbook_ids = task
            results = worker.process_batch(playbook_ids)
            result_queue.put((batch_id, results))

    except Exception as e:
        print(f"[GPU {gpu_id}] Worker crashed: {e}\n{traceback.format_exc()}")
        result_queue.put(None)


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
# Progress Reporter
# =============================================================================

class ProgressReporter:
    """Progress reporter with ETA estimation."""

    def __init__(self, total: int, report_interval: int = 100):
        self.total = total
        self.report_interval = report_interval
        self.start_time = time.time()
        self.processed = 0

    def update(self, count: int = 1) -> bool:
        """Update progress and return True if should report."""
        self.processed += count
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
    num_gpus: int = 8,
    max_memory_per_gpu: int = 20,
    batch_size: int = 16,
    report_interval: int = 100
) -> Dict:
    """
    Main analysis function with multi-GPU parallel inference.

    Args:
        playbook_path: Path to playbooks JSON file
        checkpoint_path: Path to HF checkpoint directory
        num_samples: Number of samples to analyze (0 = all)
        output_path: Output path for trace JSON
        max_rl_steps: Maximum RL steps filter (999 = no filter)
        seed: Random seed for sampling
        temperature: Sampling temperature
        num_gpus: Number of GPUs to use
        max_memory_per_gpu: Max memory per GPU in GB
        batch_size: Number of playbooks per batch
        report_interval: Progress report interval

    Returns:
        Dictionary with analysis results
    """
    logger.info("=" * 70)
    logger.info("Customer Service Agent - Bad Case Analysis (Multi-GPU Parallel)")
    logger.info("=" * 70)
    logger.info(f"Playbook: {playbook_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Num GPUs: {num_gpus}")
    logger.info(f"Max memory per GPU: {max_memory_per_gpu}GB")
    logger.info(f"Max RL steps filter: {max_rl_steps if max_rl_steps < 999 else 'No limit'}")
    logger.info(f"Num samples: {num_samples if num_samples > 0 else 'All'}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 70)

    # Load playbooks
    with open(playbook_path, 'r', encoding='utf-8') as f:
        all_playbooks = json.load(f)
    logger.info(f"Loaded {len(all_playbooks)} total playbooks")

    # Filter playbooks
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
    logger.info(f"Will process {total_to_process} playbooks with {num_gpus} GPUs")

    # Split playbooks into batches
    playbook_ids = [p['playbook_id'] for p in sample_playbooks]

    batches = []
    for i in range(0, len(playbook_ids), batch_size):
        batches.append(playbook_ids[i:i + batch_size])

    logger.info(f"Created {len(batches)} batches (batch_size={batch_size})")

    # Create queues for multiprocessing
    mp_ctx = mp.get_context('spawn')
    task_queue = mp_ctx.Queue()
    result_queue = mp_ctx.Queue()

    # Start worker processes
    workers = []
    for gpu_id in range(num_gpus):
        p = mp_ctx.Process(
            target=worker_process,
            args=(
                gpu_id,
                task_queue,
                result_queue,
                checkpoint_path,
                playbook_path,
                max_memory_per_gpu,
                temperature,
                30  # max_steps
            )
        )
        p.start()
        workers.append(p)
        logger.info(f"Started worker for GPU {gpu_id}")

    # Give workers time to initialize
    logger.info("Waiting for workers to initialize...")
    time.sleep(10)

    # Submit all tasks
    for batch_id, batch in enumerate(batches):
        task_queue.put((batch_id, batch))

    # Add poison pills
    for _ in range(num_gpus):
        task_queue.put(None)

    logger.info("All tasks submitted, waiting for results...")

    # Collect results
    progress = ProgressReporter(total_to_process, report_interval)

    results_by_batch = {}
    completed_workers = 0

    while completed_workers < num_gpus:
        result = result_queue.get()

        if result is None:
            completed_workers += 1
            continue

        batch_id, batch_results = result
        results_by_batch[batch_id] = batch_results

        # Count processed
        processed = sum(len(r.get('trace', {}).get('action_history', []))
                       for r in batch_results if r.get('success'))

        if progress.update(len(batch_results)):
            logger.info(progress.get_progress_line())

    # Wait for all workers to finish
    for p in workers:
        p.join()

    logger.info("All workers finished, aggregating results...")

    # Aggregate results
    all_traces = []
    action_error_counter = Counter()
    scenario_stats = defaultdict(lambda: {'total': 0, 'won': 0, 'failed': 0, 'errors': 0})
    length_stats = init_length_breakdown()

    for batch_id in sorted(results_by_batch.keys()):
        for result in results_by_batch[batch_id]:
            if result.get('success'):
                trace_dict = result['trace']
                trace = dict_to_trace(trace_dict)
                all_traces.append(trace)

                # Update statistics
                scenario = trace.scenario
                scenario_stats[scenario]['total'] += 1
                if trace.won:
                    scenario_stats[scenario]['won'] += 1
                else:
                    scenario_stats[scenario]['failed'] += 1
                    for err in trace.error_actions:
                        action_error_counter[err['action']] += 1
                        scenario_stats[scenario]['errors'] += 1

                # Length stats
                length_bucket = get_length_bucket(trace.rl_steps)
                length_stats[length_bucket]['total'] += 1
                if trace.won:
                    length_stats[length_bucket]['won'] += 1
                else:
                    length_stats[length_bucket]['failed'] += 1
                    length_stats[length_bucket]['errors'] += len(trace.error_actions)

    # Filter bad cases (failures only)
    bad_cases = [t for t in all_traces if not t.won]

    # Calculate success rate
    total_processed = len(all_traces)
    total_failures = len(bad_cases)
    success_rate = (total_processed - total_failures) / total_processed * 100 if total_processed > 0 else 0

    logger.info(f"Processed {total_processed} episodes, {total_failures} failures ({success_rate:.1f}% success)")

    # Prepare output
    trace_data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'playbook_path': playbook_path,
            'checkpoint_path': checkpoint_path,
            'num_samples': num_samples,
            'max_rl_steps': max_rl_steps,
            'temperature': temperature,
            'num_gpus': num_gpus,
            'max_memory_per_gpu': max_memory_per_gpu,
            'batch_size': batch_size,
            'seed': seed
        },
        'summary': {
            'total_episodes': total_processed,
            'total_failures': total_failures,
            'success_rate': f"{success_rate:.2f}%",
            'top_error_actions': action_error_counter.most_common(10),
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
        'bad_cases': [trace_to_dict(bc) for bc in bad_cases]
    }

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trace_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_path}")

    # Print summary
    print_summary(trace_data)

    return trace_data


def print_summary(data: Dict):
    """Print detailed analysis summary."""
    summary = data['summary']
    scenario_breakdown = data['scenario_breakdown']
    length_breakdown = data['length_breakdown']

    logger.info("")
    logger.info("=" * 70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total episodes analyzed: {summary['total_episodes']}")
    logger.info(f"Total failures: {summary['total_failures']}")
    logger.info(f"Overall success rate: {summary['success_rate']}")
    logger.info("")

    # Scenario breakdown
    logger.info("-" * 70)
    logger.info("SCENARIO BREAKDOWN")
    logger.info("-" * 70)
    logger.info(f"{'Scenario':<15} {'Total':>8} {'Won':>8} {'Failed':>8} {'Success%':>10} {'Errors':>8}")
    logger.info("-" * 70)
    for scenario, stats in sorted(scenario_breakdown.items()):
        logger.info(f"{scenario:<15} {stats['total']:>8} {stats['won']:>8} {stats['failed']:>8} {stats['success_rate']:>10} {stats['total_errors']:>8}")
    logger.info("")

    # Length breakdown
    logger.info("-" * 70)
    logger.info("DIALOGUE LENGTH BREAKDOWN")
    logger.info("-" * 70)
    logger.info(f"{'Length':<15} {'Total':>8} {'Won':>8} {'Failed':>8} {'Success%':>10} {'Errors':>8}")
    logger.info("-" * 70)
    for bucket, stats in length_breakdown.items():
        if stats['total'] > 0:
            logger.info(f"{bucket:<15} {stats['total']:>8} {stats['won']:>8} {stats['failed']:>8} {stats['success_rate']:>10} {stats['total_errors']:>8}")
    logger.info("")

    # Top errors
    logger.info("-" * 70)
    logger.info("TOP 10 ERROR ACTIONS")
    logger.info("-" * 70)
    for action, count in summary['top_error_actions']:
        logger.info(f"  {action}: {count} errors")
    logger.info("")

    logger.info("=" * 70)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze bad cases with Multi-GPU Parallel Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full dataset evaluation with 8 GPUs (default)
  python scripts/analyze_bad_cases.py \\
      --checkpoint_path outputs/hf_checkpoints/epoch_40 \\
      --num_gpus 8 \\
      --max_memory_per_gpu 20

  # Single GPU mode
  python scripts/analyze_bad_cases.py \\
      --checkpoint_path outputs/hf_checkpoints/epoch_40 \\
      --num_gpus 1

  # Sample 100 playbooks for quick testing
  python scripts/analyze_bad_cases.py \\
      --checkpoint_path outputs/hf_checkpoints/epoch_40 \\
      --num_samples 100
        """
    )
    parser.add_argument('--playbook_path', type=str, default='outputs/playbooks_all.json',
                        help='Path to playbooks JSON file')
    parser.add_argument('--checkpoint_path', type=str,
                        default='/home/bo.li/data/SkillRL/skillrl_models/customer_service/epoch_160',
                        help='Path to HF checkpoint directory')
    parser.add_argument('--num_samples', type=int, default=0,
                        help='Number of samples to analyze (0 = all)')
    parser.add_argument('--max_rl_steps', type=int, default=999,
                        help='Maximum RL steps filter (999 = no filter)')
    parser.add_argument('--output_path', type=str, default='outputs/all_bad_cases_trace.json',
                        help='Output path for trace JSON')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--temperature', type=float, default=0.4,
                        help='Sampling temperature')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of GPUs to use for parallel inference')
    parser.add_argument('--max_memory_per_gpu', type=int, default=20,
                        help='Maximum memory per GPU in GB (to not interfere with training)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of playbooks per batch per GPU')
    parser.add_argument('--report_interval', type=int, default=100,
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
        num_gpus=args.num_gpus,
        max_memory_per_gpu=args.max_memory_per_gpu,
        batch_size=args.batch_size,
        report_interval=args.report_interval
    )


if __name__ == '__main__':
    # Required for multiprocessing with CUDA
    mp.set_start_method('spawn', force=True)
    main()