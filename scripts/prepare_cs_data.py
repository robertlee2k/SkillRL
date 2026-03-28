#!/usr/bin/env python
"""
Industrial-grade Customer Service data preparation for veRL/GiGPO training.

Features:
  - O(1) rl_steps filtering: Discard playbooks exceeding max_rl_steps threshold
  - Stratified sampling: Maintain scenario distribution across train/val splits
  - Auto drop-last alignment: Ensure dataset sizes divisible by batch sizes
  - Fail-fast assertions: Validate data integrity before parquet generation
  - Rich logging: Distribution comparison, statistics, and audit trail

Parameters (ratio-driven):
  --val_ratio        Validation split ratio (default: 0.15)
  --train_batch_size Training batch size for drop-last alignment (default: 64)
  --val_batch_size   Validation batch size for drop-last alignment (default: 128)
  --stratify_by      Field for stratified sampling (default: 'scenario')
  --max_rl_steps     Maximum allowed RL steps, excess discarded (default: 20)

Usage:
    python scripts/prepare_cs_data.py \
        --playbook_path outputs/playbooks_all.json \
        --output_dir ~/data/verl-agent/customer_service \
        --val_ratio 0.15 \
        --train_batch_size 64 \
        --val_batch_size 128 \
        --max_rl_steps 20

Output:
    train.parquet: Training data (size divisible by train_batch_size)
    test.parquet:  Validation data (size divisible by val_batch_size)
    report.json:   Preparation audit report with statistics
"""

import os
import json
import argparse
import random
import logging
from typing import List, Dict, Any
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# System prompt for customer service agent
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


class DataPreparationReport:
    """Audit report for data preparation process."""

    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.input_path = ""
        self.output_dir = ""
        self.params = {}

        # Stage 1: Input stats
        self.input_total = 0
        self.input_scenario_dist = {}
        self.input_rl_steps_stats = {}

        # Stage 2: After rl_steps filter
        self.after_filter_total = 0
        self.after_filter_discarded = 0
        self.after_filter_scenario_dist = {}
        self.after_filter_rl_steps_stats = {}

        # Stage 3: After stratified split
        self.train_total = 0
        self.val_total = 0
        self.train_scenario_dist = {}
        self.val_scenario_dist = {}

        # Stage 4: After drop-last alignment
        self.train_final = 0
        self.val_final = 0
        self.train_discarded = 0
        self.val_discarded = 0

        # Stage 5: Final output
        self.train_path = ""
        self.val_path = ""
        self.verification_passed = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'input_path': self.input_path,
            'output_dir': self.output_dir,
            'parameters': self.params,

            'stage_1_input': {
                'total': self.input_total,
                'scenario_distribution': self.input_scenario_dist,
                'rl_steps_stats': self.input_rl_steps_stats,
            },

            'stage_2_rl_steps_filter': {
                'total_after': self.after_filter_total,
                'discarded': self.after_filter_discarded,
                'retention_rate': f"{self.after_filter_total / self.input_total * 100:.2f}%" if self.input_total > 0 else "N/A",
                'scenario_distribution': self.after_filter_scenario_dist,
                'rl_steps_stats': self.after_filter_rl_steps_stats,
            },

            'stage_3_stratified_split': {
                'train_total': self.train_total,
                'val_total': self.val_total,
                'train_scenario_distribution': self.train_scenario_dist,
                'val_scenario_distribution': self.val_scenario_dist,
                'distribution_match': self._compute_distribution_match(),
            },

            'stage_4_drop_last_alignment': {
                'train_final': self.train_final,
                'val_final': self.val_final,
                'train_discarded': self.train_discarded,
                'val_discarded': self.val_discarded,
                'train_batches': self.train_final // self.params.get('train_batch_size', 64),
                'val_batches': self.val_final // self.params.get('val_batch_size', 128),
            },

            'stage_5_output': {
                'train_path': self.train_path,
                'val_path': self.val_path,
                'verification_passed': self.verification_passed,
            }
        }

    def _compute_distribution_match(self) -> Dict[str, Any]:
        """Compute how well train/val distributions match the original."""
        if not self.after_filter_scenario_dist:
            return {'error': 'No distribution data'}

        original_dist = self.after_filter_scenario_dist
        train_dist = self.train_scenario_dist
        val_dist = self.val_scenario_dist

        # Compute KL divergence-like metric for each split
        def relative_error(original: Dict, split: Dict) -> Dict[str, float]:
            errors = {}
            total_original = sum(original.values())
            total_split = sum(split.values())

            for key in original:
                orig_pct = original[key] / total_original if total_original > 0 else 0
                split_pct = split.get(key, 0) / total_split if total_split > 0 else 0
                errors[key] = abs(orig_pct - split_pct) * 100
            return errors

        train_errors = relative_error(original_dist, train_dist)
        val_errors = relative_error(original_dist, val_dist)

        return {
            'train_relative_error_pct': train_errors,
            'val_relative_error_pct': val_errors,
            'max_train_error': max(train_errors.values()) if train_errors else 0,
            'max_val_error': max(val_errors.values()) if val_errors else 0,
        }

    def save(self, path: str) -> None:
        """Save report to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {path}")


def load_playbooks(playbook_path: str) -> List[Dict[str, Any]]:
    """Load playbooks from JSON file."""
    with open(playbook_path, 'r', encoding='utf-8') as f:
        playbooks = json.load(f)
    logger.info(f"Loaded {len(playbooks)} playbooks from {playbook_path}")
    return playbooks


def compute_scenario_distribution(playbooks: List[Dict]) -> Dict[str, int]:
    """Compute scenario distribution count."""
    counter = Counter(p.get('scenario', 'unknown') for p in playbooks)
    return dict(counter)


def compute_rl_steps_stats(playbooks: List[Dict]) -> Dict[str, Any]:
    """Compute RL steps statistics."""
    rl_steps_values = [p.get('rl_steps', 0) for p in playbooks]

    if not rl_steps_values:
        return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'missing_count': 0}

    # Handle None values
    valid_values = [v for v in rl_steps_values if v is not None]
    missing_count = len(rl_steps_values) - len(valid_values)

    if not valid_values:
        return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'missing_count': missing_count}

    return {
        'min': int(min(valid_values)),
        'max': int(max(valid_values)),
        'mean': float(np.mean(valid_values)),
        'median': float(np.median(valid_values)),
        'missing_count': missing_count,
        'distribution': dict(Counter(valid_values)),
    }


def filter_by_rl_steps(
    playbooks: List[Dict],
    max_rl_steps: int,
    report: DataPreparationReport
) -> List[Dict]:
    """
    Filter playbooks by max_rl_steps threshold (O(1) operation per playbook).

    Discards playbooks where rl_steps > max_rl_steps.
    Playbooks with missing rl_steps are kept (conservative approach).
    """
    logger.info(f"Stage 2: Filtering by rl_steps <= {max_rl_steps}")

    filtered = []
    discarded = []

    for p in playbooks:
        rl_steps = p.get('rl_steps')
        # Keep if rl_steps is missing (None) or within threshold
        if rl_steps is None or rl_steps <= max_rl_steps:
            filtered.append(p)
        else:
            discarded.append(p)

    report.after_filter_total = len(filtered)
    report.after_filter_discarded = len(discarded)
    report.after_filter_scenario_dist = compute_scenario_distribution(filtered)
    report.after_filter_rl_steps_stats = compute_rl_steps_stats(filtered)

    logger.info(f"  ✓ Retained: {len(filtered)} playbooks")
    logger.info(f"  ✗ Discarded: {len(discarded)} playbooks (rl_steps > {max_rl_steps})")

    if discarded:
        discard_dist = compute_scenario_distribution(discarded)
        logger.info(f"  Discarded distribution: {discard_dist}")

    # Fail-fast: warn if retention rate is too low
    retention_rate = len(filtered) / len(playbooks) if playbooks else 0
    if retention_rate < 0.8:
        logger.warning(f"  ⚠ Low retention rate: {retention_rate * 100:.1f}%")

    return filtered


def stratified_split(
    playbooks: List[Dict],
    val_ratio: float,
    stratify_by: str,
    seed: int,
    report: DataPreparationReport
) -> tuple[List[Dict], List[Dict]]:
    """
    Perform stratified split to maintain distribution across train/val.

    Args:
        playbooks: List of playbook dicts
        val_ratio: Validation ratio (e.g., 0.15)
        stratify_by: Field name for stratification (e.g., 'scenario')
        seed: Random seed
        report: Report object to populate

    Returns:
        (train_playbooks, val_playbooks)
    """
    logger.info(f"Stage 3: Stratified split by '{stratify_by}' with val_ratio={val_ratio}")

    random.seed(seed)

    # Group by stratification key
    groups: Dict[str, List[Dict]] = {}
    for p in playbooks:
        key = p.get(stratify_by, 'unknown')
        if key not in groups:
            groups[key] = []
        groups[key].append(p)

    train_playbooks = []
    val_playbooks = []

    # Split each group proportionally
    for key, group in sorted(groups.items()):
        group_size = len(group)
        val_size = int(group_size * val_ratio)

        # Shuffle within group
        shuffled = group.copy()
        random.shuffle(shuffled)

        # Split
        val_group = shuffled[:val_size]
        train_group = shuffled[val_size:]

        train_playbooks.extend(train_group)
        val_playbooks.extend(val_group)

        logger.info(f"  {key}: {group_size} → train={len(train_group)}, val={len(val_group)}")

    report.train_total = len(train_playbooks)
    report.val_total = len(val_playbooks)
    report.train_scenario_dist = compute_scenario_distribution(train_playbooks)
    report.val_scenario_dist = compute_scenario_distribution(val_playbooks)

    logger.info(f"  ✓ Train total: {len(train_playbooks)}")
    logger.info(f"  ✓ Val total: {len(val_playbooks)}")

    # Verify distribution match
    match = report._compute_distribution_match()
    max_train_error = match.get('max_train_error', 0)
    max_val_error = match.get('max_val_error', 0)

    if max_train_error > 5 or max_val_error > 5:
        logger.warning(f"  ⚠ Distribution mismatch: train error={max_train_error:.1f}%, val error={max_val_error:.1f}%")
    else:
        logger.info(f"  ✓ Distribution match: train error={max_train_error:.1f}%, val error={max_val_error:.1f}%")

    return train_playbooks, val_playbooks


def drop_last_align(
    playbooks: List[Dict],
    batch_size: int,
    seed: int,
    split_name: str,
    report: DataPreparationReport
) -> List[Dict]:
    """
    Align dataset size to be divisible by batch_size (drop-last alignment).

    Removes excess samples to ensure every batch is complete.
    """
    logger.info(f"Stage 4 ({split_name}): Drop-last alignment to batch_size={batch_size}")

    current_size = len(playbooks)
    remainder = current_size % batch_size
    aligned_size = current_size - remainder

    if remainder == 0:
        logger.info(f"  ✓ Already aligned: {current_size} samples ({current_size // batch_size} batches)")
        return playbooks

    # Remove excess samples randomly
    random.seed(seed)
    excess_indices = random.sample(range(current_size), remainder)
    aligned = [p for i, p in enumerate(playbooks) if i not in excess_indices]

    logger.info(f"  ✓ Aligned: {aligned_size} samples ({aligned_size // batch_size} batches)")
    logger.info(f"  ✗ Discarded: {remainder} excess samples")

    # Update report
    if split_name == 'train':
        report.train_final = aligned_size
        report.train_discarded = remainder
    else:
        report.val_final = aligned_size
        report.val_discarded = remainder

    return aligned


def create_initial_prompt(playbook: Dict[str, Any]) -> str:
    """
    Create the initial prompt for a playbook episode.
    """
    scenario = playbook.get('scenario', 'unknown')
    subtype = playbook.get('subtype', 'general')
    nodes = playbook.get('nodes', {})

    # Get root node buyer text
    root_node = nodes.get('root', {})
    buyer_text = root_node.get('buyer_text', '')

    scenario_desc = {
        'presale': '售前咨询',
        'logistics': '物流查询',
        'aftersale': '售后服务',
        'unknown': '客服咨询'
    }.get(scenario, '客服咨询')

    prompt = f"""## 场景信息
场景类型: {scenario_desc} ({scenario})
子类型: {subtype}

## 买家消息
{buyer_text}

## 任务
请分析买家的需求，并选择合适的客服动作进行回应。记住：
1. 如果买家情绪不好（angry），优先安抚情绪
2. 选择正确的动作推进对话
3. 避免触发不必要的 fallback"""

    return prompt


def playbook_to_record(playbook: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Convert a playbook to a training record.
    """
    playbook_id = playbook.get('playbook_id', f'unknown_{idx}')
    scenario = playbook.get('scenario', 'unknown')
    subtype = playbook.get('subtype', 'general')
    session_id = playbook.get('session_id', '')
    business_outcome = playbook.get('business_outcome', {})
    rl_steps = playbook.get('rl_steps')
    effective_turn_count = playbook.get('effective_turn_count')

    initial_prompt = create_initial_prompt(playbook)

    record = {
        'data_source': 'customer_service',
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': initial_prompt}
        ],
        'ability': 'agent',
        'extra_info': {
            'playbook_id': playbook_id,
            'session_id': session_id,
            'scenario': scenario,
            'subtype': subtype,
            'business_outcome': business_outcome,
            'rl_steps': rl_steps,
            'effective_turn_count': effective_turn_count,
            'index': idx,
        }
    }

    return record


def verify_output(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_batch_size: int,
    val_batch_size: int,
    report: DataPreparationReport
) -> bool:
    """
    Fail-fast verification of output data integrity.
    """
    logger.info("Stage 5: Output verification")

    errors = []

    # Check 1: Batch alignment
    if len(train_df) % train_batch_size != 0:
        errors.append(f"Train size {len(train_df)} not divisible by batch_size {train_batch_size}")
    else:
        logger.info(f"  ✓ Train aligned: {len(train_df)} samples = {len(train_df) // train_batch_size} batches")

    if len(val_df) % val_batch_size != 0:
        errors.append(f"Val size {len(val_df)} not divisible by batch_size {val_batch_size}")
    else:
        logger.info(f"  ✓ Val aligned: {len(val_df)} samples = {len(val_df) // val_batch_size} batches")

    # Check 2: Required columns
    required_cols = ['data_source', 'prompt', 'ability', 'extra_info']
    for col in required_cols:
        if col not in train_df.columns:
            errors.append(f"Train missing required column: {col}")
        if col not in val_df.columns:
            errors.append(f"Val missing required column: {col}")

    if not errors:
        logger.info(f"  ✓ All required columns present")

    # Check 3: Prompt format
    sample_prompt = train_df['prompt'].iloc[0]
    if not isinstance(sample_prompt, list) or len(sample_prompt) < 2:
        errors.append(f"Invalid prompt format: expected list with >=2 elements")
    else:
        logger.info(f"  ✓ Prompt format valid")

    # Check 4: No duplicate playbook_ids
    train_ids = train_df['extra_info'].apply(lambda x: x.get('playbook_id')).tolist()
    val_ids = val_df['extra_info'].apply(lambda x: x.get('playbook_id')).tolist()

    train_duplicates = len(train_ids) - len(set(train_ids))
    val_duplicates = len(val_ids) - len(set(val_ids))
    cross_overlap = len(set(train_ids) & set(val_ids))

    if train_duplicates > 0:
        errors.append(f"Train has {train_duplicates} duplicate playbook_ids")
    if val_duplicates > 0:
        errors.append(f"Val has {val_duplicates} duplicate playbook_ids")
    if cross_overlap > 0:
        errors.append(f"Train/Val overlap: {cross_overlap} shared playbook_ids")

    if not any([train_duplicates, val_duplicates, cross_overlap]):
        logger.info(f"  ✓ No duplicates or overlap")

    # Check 5: RL steps within threshold (from extra_info)
    train_rl_steps = train_df['extra_info'].apply(lambda x: x.get('rl_steps', 0)).tolist()
    val_rl_steps = val_df['extra_info'].apply(lambda x: x.get('rl_steps', 0)).tolist()

    train_exceed = sum(1 for s in train_rl_steps if s is not None and s > 20)  # Default threshold
    val_exceed = sum(1 for s in val_rl_steps if s is not None and s > 20)

    if train_exceed > 0:
        errors.append(f"Train has {train_exceed} samples exceeding rl_steps threshold")
    if val_exceed > 0:
        errors.append(f"Val has {val_exceed} samples exceeding rl_steps threshold")

    if not errors:
        logger.info(f"  ✓ All rl_steps within threshold")

    # Report result
    if errors:
        logger.error("  ✗ Verification FAILED:")
        for e in errors:
            logger.error(f"    - {e}")
        report.verification_passed = False
        return False

    logger.info("  ✓ All verification checks PASSED")
    report.verification_passed = True
    return True


def prepare_data(
    playbook_path: str,
    output_dir: str,
    val_ratio: float,
    train_batch_size: int,
    val_batch_size: int,
    stratify_by: str,
    max_rl_steps: int,
    seed: int = 42
) -> DataPreparationReport:
    """
    Industrial-grade data preparation pipeline.

    Pipeline stages:
      1. Load and analyze input playbooks
      2. Filter by max_rl_steps threshold (O(1) per playbook)
      3. Stratified split maintaining scenario distribution
      4. Drop-last alignment to batch sizes
      5. Verify output integrity and save parquet
    """
    report = DataPreparationReport()
    report.input_path = playbook_path
    report.output_dir = output_dir
    report.params = {
        'val_ratio': val_ratio,
        'train_batch_size': train_batch_size,
        'val_batch_size': val_batch_size,
        'stratify_by': stratify_by,
        'max_rl_steps': max_rl_steps,
        'seed': seed,
    }

    logger.info("=" * 60)
    logger.info("Industrial-Grade Data Preparation Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input: {playbook_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Params: val_ratio={val_ratio}, train_bs={train_batch_size}, val_bs={val_batch_size}")
    logger.info(f"       stratify_by={stratify_by}, max_rl_steps={max_rl_steps}, seed={seed}")
    logger.info("=" * 60)

    # ===== Stage 1: Load and analyze input =====
    logger.info("Stage 1: Loading and analyzing input")
    playbooks = load_playbooks(playbook_path)

    report.input_total = len(playbooks)
    report.input_scenario_dist = compute_scenario_distribution(playbooks)
    report.input_rl_steps_stats = compute_rl_steps_stats(playbooks)

    logger.info(f"  Input distribution: {report.input_scenario_dist}")
    logger.info(f"  RL steps stats: min={report.input_rl_steps_stats['min']}, "
                f"max={report.input_rl_steps_stats['max']}, "
                f"mean={report.input_rl_steps_stats['mean']:.1f}")

    # ===== Stage 2: Filter by rl_steps =====
    filtered = filter_by_rl_steps(playbooks, max_rl_steps, report)

    # Fail-fast: Check minimum data after filtering
    min_required = train_batch_size + val_batch_size  # At least one batch each
    if len(filtered) < min_required:
        raise ValueError(
            f"After filtering, only {len(filtered)} playbooks available. "
            f"Need at least {min_required} for one batch of train and val. "
            f"Consider lowering max_rl_steps={max_rl_steps} or increasing playbook count."
        )

    # ===== Stage 3: Stratified split =====
    train_playbooks, val_playbooks = stratified_split(
        filtered, val_ratio, stratify_by, seed, report
    )

    # Fail-fast: Check minimum per split
    if len(train_playbooks) < train_batch_size:
        raise ValueError(
            f"Train split has {len(train_playbooks)} samples, "
            f"less than train_batch_size={train_batch_size}"
        )
    if len(val_playbooks) < val_batch_size:
        raise ValueError(
            f"Val split has {len(val_playbooks)} samples, "
            f"less than val_batch_size={val_batch_size}"
        )

    # ===== Stage 4: Drop-last alignment =====
    train_aligned = drop_last_align(train_playbooks, train_batch_size, seed, 'train', report)
    val_aligned = drop_last_align(val_playbooks, val_batch_size, seed, 'val', report)

    # ===== Stage 5: Convert to records and verify =====
    logger.info("Stage 5: Converting to records")

    train_records = [playbook_to_record(p, i) for i, p in enumerate(train_aligned)]
    val_records = [playbook_to_record(p, i) for i, p in enumerate(val_aligned)]

    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)

    logger.info(f"  ✓ Train: {len(train_df)} records")
    logger.info(f"  ✓ Val: {len(val_df)} records")

    # Verify before saving
    if not verify_output(train_df, val_df, train_batch_size, val_batch_size, report):
        raise ValueError("Output verification failed. See errors above.")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save parquet files
    train_path = os.path.join(output_dir, 'train.parquet')
    val_path = os.path.join(output_dir, 'test.parquet')

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    report.train_path = train_path
    report.val_path = val_path

    logger.info(f"  ✓ Saved train to {train_path}")
    logger.info(f"  ✓ Saved val to {val_path}")

    # Save report
    report_path = os.path.join(output_dir, 'report.json')
    report.save(report_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("Pipeline Summary")
    logger.info("=" * 60)
    logger.info(f"Input:     {report.input_total} playbooks")
    logger.info(f"Filtered:  {report.after_filter_total} (discarded {report.after_filter_discarded})")
    logger.info(f"Train:     {report.train_final} samples ({report.train_final // train_batch_size} batches)")
    logger.info(f"Val:       {report.val_final} samples ({report.val_final // val_batch_size} batches)")
    logger.info(f"Total kept: {report.train_final + report.val_final}")
    logger.info("=" * 60)

    # Print sample
    logger.info("\nSample record:")
    sample = train_records[0]
    logger.info(f"  data_source: {sample['data_source']}")
    logger.info(f"  ability: {sample['ability']}")
    logger.info(f"  extra_info: {sample['extra_info']}")
    prompt_preview = sample['prompt'][1]['content'][:100] if len(sample['prompt']) > 1 else "N/A"
    logger.info(f"  prompt preview: {prompt_preview}...")

    return report


def main():
    parser = argparse.ArgumentParser(
        description='Industrial-grade Customer Service data preparation for veRL training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default usage
    python scripts/prepare_cs_data.py

    # Custom parameters
    python scripts/prepare_cs_data.py \
        --playbook_path outputs/playbooks_all.json \
        --output_dir ~/data/verl-agent/customer_service \
        --val_ratio 0.15 \
        --train_batch_size 64 \
        --val_batch_size 128 \
        --max_rl_steps 20

Output files:
    train.parquet: Training data (aligned to train_batch_size)
    test.parquet:  Validation data (aligned to val_batch_size)
    report.json:   Preparation audit report
        """
    )

    parser.add_argument('--playbook_path', type=str,
                        default='outputs/playbooks_all.json',
                        help='Path to playbooks JSON file')
    parser.add_argument('--output_dir', type=str,
                        default='~/data/verl-agent/customer_service',
                        help='Output directory for parquet files')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation split ratio (default: 0.15)')
    parser.add_argument('--train_batch_size', type=int, default=64,
                        help='Training batch size for alignment (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='Validation batch size for alignment (default: 128)')
    parser.add_argument('--stratify_by', type=str, default='scenario',
                        help='Field for stratified sampling (default: scenario)')
    parser.add_argument('--max_rl_steps', type=int, default=20,
                        help='Maximum RL steps threshold, excess discarded (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Expand home directory
    output_dir = os.path.expanduser(args.output_dir)

    # Validate parameters
    if args.val_ratio <= 0 or args.val_ratio >= 1:
        raise ValueError(f"val_ratio must be between 0 and 1, got {args.val_ratio}")

    if args.train_batch_size <= 0:
        raise ValueError(f"train_batch_size must be positive, got {args.train_batch_size}")

    if args.val_batch_size <= 0:
        raise ValueError(f"val_batch_size must be positive, got {args.val_batch_size}")

    if args.max_rl_steps <= 0:
        raise ValueError(f"max_rl_steps must be positive, got {args.max_rl_steps}")

    # Run pipeline
    try:
        report = prepare_data(
            playbook_path=args.playbook_path,
            output_dir=output_dir,
            val_ratio=args.val_ratio,
            train_batch_size=args.train_batch_size,
            val_batch_size=args.val_batch_size,
            stratify_by=args.stratify_by,
            max_rl_steps=args.max_rl_steps,
            seed=args.seed
        )
        logger.info("\n✓ Data preparation completed successfully!")

    except ValueError as e:
        logger.error(f"\n✗ Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()