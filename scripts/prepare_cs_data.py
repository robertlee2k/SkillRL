#!/usr/bin/env python
"""
Prepare Customer Service data for veRL/GiGPO training.

This script reads playbooks_new.json and converts them to parquet format
compatible with the verl training pipeline.

Output format (parquet):
- data_source: 'customer_service'
- prompt: [{'role': 'user', 'content': '...'}]  # Initial prompt for the agent
- ability: 'agent'
- extra_info: {'playbook_id': '...', 'scenario': '...', ...}

Usage:
    python scripts/prepare_cs_data.py \
        --playbook_path outputs/playbooks_new.json \
        --output_dir ~/data/verl-agent/customer_service \
        --train_data_size 16 \
        --val_data_size 64
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


# System prompt for customer service agent
SYSTEM_PROMPT = """你是一名专业的电商客服智能助手，擅长处理售前咨询、物流查询和售后问题。
你的目标是高效解决买家问题，促成交易或妥善处理售后，同时避免激怒买家。

你需要在每一步选择正确的服务动作。你的输出格式必须是：

<tool_call>
[你的分析过程：买家想要什么？当前对话处于什么阶段？哪个动作最合适？]
.serif:montserrat<action>skill_id</action>

可用的动作 ID 包括：
- 通用: gen_greet, gen_empathize, gen_clarify, gen_verify_order, gen_hold, gen_transfer, gen_apologize, gen_close
- 售前: pre_query_product, pre_check_stock, pre_compare, pre_recommend, pre_answer_spec, pre_check_promo, pre_guide_purchase
- 物流: log_query_status, log_query_detail, log_estimate_arrival, log_modify_address, log_contact_courier, log_delay_notify, log_lost_claim
- 售后: aft_check_policy, aft_collect_evidence, aft_initiate_refund, aft_initiate_return, aft_initiate_exchange, aft_schedule_pickup, aft_track_progress, aft_compensate, aft_reject_explain
"""


def load_playbooks(playbook_path: str) -> List[Dict[str, Any]]:
    """Load playbooks from JSON file."""
    with open(playbook_path, 'r', encoding='utf-8') as f:
        playbooks = json.load(f)
    print(f"[prepare_cs_data] Loaded {len(playbooks)} playbooks from {playbook_path}")
    return playbooks


def create_initial_prompt(playbook: Dict[str, Any]) -> str:
    """
    Create the initial prompt for a playbook episode.

    This prompt describes the scenario and gives the agent context
    for what to expect in this conversation.
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

    Args:
        playbook: The playbook dictionary
        idx: Index for the record

    Returns:
        A dictionary with keys: data_source, prompt, ability, extra_info
    """
    playbook_id = playbook.get('playbook_id', f'unknown_{idx}')
    scenario = playbook.get('scenario', 'unknown')
    subtype = playbook.get('subtype', 'general')
    session_id = playbook.get('session_id', '')
    business_outcome = playbook.get('business_outcome', {})

    # Create initial prompt
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
            'index': idx,
        }
    }

    return record


def prepare_data(
    playbook_path: str,
    output_dir: str,
    train_data_size: int,
    val_data_size: int,
    seed: int = 42
) -> None:
    """
    Prepare training and validation data from playbooks.

    Args:
        playbook_path: Path to playbooks JSON file
        output_dir: Directory to save parquet files
        train_data_size: Number of training samples
        val_data_size: Number of validation samples
        seed: Random seed for reproducibility
    """
    # Load playbooks
    playbooks = load_playbooks(playbook_path)

    # Set random seed
    random.seed(seed)

    # Shuffle playbooks
    random.shuffle(playbooks)

    # Split into train and validation
    # For training, we duplicate playbooks if needed to reach train_data_size
    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []

    # Generate training records
    train_idx = 0
    while len(train_records) < train_data_size:
        for playbook in playbooks:
            if len(train_records) >= train_data_size:
                break
            record = playbook_to_record(playbook, train_idx)
            train_records.append(record)
            train_idx += 1

    # Generate validation records
    val_idx = 0
    while len(val_records) < val_data_size:
        for playbook in playbooks:
            if len(val_records) >= val_data_size:
                break
            record = playbook_to_record(playbook, val_idx)
            val_records.append(record)
            val_idx += 1

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrames
    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)

    # Save to parquet
    train_path = os.path.join(output_dir, 'train.parquet')
    val_path = os.path.join(output_dir, 'test.parquet')

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    print(f"[prepare_cs_data] Saved {len(train_records)} training records to {train_path}")
    print(f"[prepare_cs_data] Saved {len(val_records)} validation records to {val_path}")

    # Print sample
    print(f"\n[prepare_cs_data] Sample record:")
    sample = train_records[0]
    print(f"  data_source: {sample['data_source']}")
    print(f"  ability: {sample['ability']}")
    print(f"  extra_info: {sample['extra_info']}")
    print(f"  prompt[0]: {sample['prompt'][0]['content'][:100]}...")


def main():
    parser = argparse.ArgumentParser(description='Prepare Customer Service data for veRL training')
    parser.add_argument('--playbook_path', type=str, default='outputs/playbooks_new.json',
                        help='Path to playbooks JSON file')
    parser.add_argument('--output_dir', type=str, default='~/data/verl-agent/customer_service',
                        help='Output directory for parquet files')
    parser.add_argument('--train_data_size', type=int, default=16,
                        help='Number of training samples')
    parser.add_argument('--val_data_size', type=int, default=64,
                        help='Number of validation samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Expand home directory
    output_dir = os.path.expanduser(args.output_dir)

    prepare_data(
        playbook_path=args.playbook_path,
        output_dir=output_dir,
        train_data_size=args.train_data_size,
        val_data_size=args.val_data_size,
        seed=args.seed
    )


if __name__ == '__main__':
    main()