#!/usr/bin/env python3
"""
存量数据回填脚本：为现有 Playbooks 添加 effective_turn_count 字段

核心逻辑：
- 复用 etl/cleaner.py 和 etl/aggregator.py 的清洗逻辑
- 计算经过营销过滤、QA 合并等操作后的有效对话轮次

Usage:
    python scripts/backfill_turn_counts.py \
        --input outputs/playbooks_full.json \
        --output outputs/playbooks_full_with_turns.json

    # 或者原地覆盖
    python scripts/backfill_turn_counts.py \
        --input outputs/playbooks_full.json \
        --inplace
"""

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目根目录到 Python 路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.aggregator import aggregate_turns
from etl.cleaner import validate_user_agent_alternation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_raw_sessions(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    加载原始 session 数据，构建 session_id -> session 的映射

    假设原始数据文件格式为：
    - 直接列表: [{"session_id": "...", "messages": [...]}, ...]
    - 包装格式: {"data": [{"session_id": "...", "messages": [...]}, ...]}
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict) and 'data' in raw_data:
        sessions = raw_data['data']
    elif isinstance(raw_data, list):
        sessions = raw_data
    else:
        raise ValueError(f"Unexpected JSON format in {file_path}")

    # 构建 session_id -> session 映射
    session_map = {}
    for session in sessions:
        session_id = session.get('session_id')
        if session_id:
            session_map[session_id] = session

    logger.info(f"Loaded {len(session_map)} raw sessions from {file_path}")
    return session_map


def calculate_effective_turn_count(
    messages: List[Dict[str, Any]],
    session_id: str = "unknown"
) -> Dict[str, int]:
    """
    计算有效对话轮次和 RL steps（复用清洗逻辑）

    这个函数模拟 etl/cleaner.py 中的 clean_session 逻辑：
    1. 调用 aggregate_turns 进行消息聚合（过滤 MARKETING，合并连续同角色）
    2. 处理以 Agent 开头的 session（删除开头的 Agent turns）
    3. 返回最终的 turns 数量和 RL steps

    Args:
        messages: 原始消息列表
        session_id: Session ID（用于日志）

    Returns:
        Dict with 'effective_turn_count' and 'rl_steps'
    """
    # Step 1: 聚合消息（过滤 MARKETING，合并连续同角色）
    aggregated = aggregate_turns(messages)
    turns = aggregated['turns']

    # Step 2: 处理以 Agent 开头的 session
    if turns and turns[0]['role'] == 'Agent':
        # 找到第一个 User turn
        first_user_idx = None
        for i, turn in enumerate(turns):
            if turn['role'] == 'User':
                first_user_idx = i
                break

        if first_user_idx is not None:
            # 删除开头的 Agent turns
            turns = turns[first_user_idx:]
        else:
            # 全是 Agent 消息
            return {'effective_turn_count': 0, 'rl_steps': 0}

    # Step 3: 验证 User-Agent 交替
    if not validate_user_agent_alternation(turns):
        logger.warning(f"[{session_id}] Invalid turn alternation")
        # 仍然返回数量，但记录警告

    # Step 4: 计算 RL steps = User turns 数量
    user_turn_count = sum(1 for t in turns if t['role'] == 'User')

    return {
        'effective_turn_count': len(turns),
        'rl_steps': user_turn_count
    }


def backfill_playbooks(
    playbooks: List[Dict[str, Any]],
    raw_session_map: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    为 playbooks 回填 effective_turn_count 字段

    Args:
        playbooks: 现有 playbook 列表
        raw_session_map: session_id -> raw session 的映射

    Returns:
        更新后的 playbook 列表
    """
    updated_playbooks = []
    stats = {
        'total': 0,
        'found_raw': 0,
        'not_found': 0,
        'already_has_field': 0,
    }
    turn_counts = []
    rl_steps_counts = []

    for pb in playbooks:
        stats['total'] += 1
        session_id = pb.get('session_id')

        # 检查是否已有字段
        if 'effective_turn_count' in pb and 'rl_steps' in pb:
            stats['already_has_field'] += 1
            turn_counts.append(pb['effective_turn_count'])
            rl_steps_counts.append(pb['rl_steps'])
            updated_playbooks.append(pb)
            continue

        # 查找原始 session
        raw_session = raw_session_map.get(session_id)
        if raw_session is None:
            stats['not_found'] += 1
            logger.warning(f"[{session_id}] Raw session not found, skipping")
            updated_playbooks.append(pb)
            continue

        stats['found_raw'] += 1

        # 计算有效轮次和 RL steps
        messages = raw_session.get('messages', [])
        counts = calculate_effective_turn_count(messages, session_id)

        # 更新 playbook
        pb_updated = pb.copy()
        pb_updated['effective_turn_count'] = counts['effective_turn_count']
        pb_updated['rl_steps'] = counts['rl_steps']
        updated_playbooks.append(pb_updated)
        turn_counts.append(counts['effective_turn_count'])
        rl_steps_counts.append(counts['rl_steps'])

    # 打印统计信息
    logger.info("=" * 60)
    logger.info("回填统计:")
    logger.info(f"  总 Playbooks: {stats['total']}")
    logger.info(f"  已有字段: {stats['already_has_field']}")
    logger.info(f"  找到原始数据: {stats['found_raw']}")
    logger.info(f"  未找到原始数据: {stats['not_found']}")

    if turn_counts:
        sorted_counts = sorted(turn_counts)
        n = len(sorted_counts)
        logger.info("")
        logger.info("有效对话轮次 (turns) 统计:")
        logger.info(f"  最小值: {sorted_counts[0]}")
        logger.info(f"  最大值: {sorted_counts[-1]}")
        logger.info(f"  平均值: {statistics.mean(sorted_counts):.1f}")
        logger.info(f"  P90: {sorted_counts[int(n * 0.90)]}")
        logger.info(f"  P95: {sorted_counts[int(n * 0.95)]}")
        logger.info(f"  P99: {sorted_counts[int(n * 0.99)]}")

        # 统计超长对话（> 20 轮 = 40 turns）
        over_40 = sum(1 for c in turn_counts if c > 40)
        over_20 = sum(1 for c in turn_counts if c > 20)
        logger.info(f"")
        logger.info(f"  超过 40 turns (>20 RL steps): {over_40} ({over_40/len(turn_counts)*100:.1f}%)")
        logger.info(f"  超过 20 turns (>10 RL steps): {over_20} ({over_20/len(turn_counts)*100:.1f}%)")

    if rl_steps_counts:
        sorted_rl_steps = sorted(rl_steps_counts)
        n = len(sorted_rl_steps)
        logger.info("")
        logger.info("RL steps (Agent action 数) 统计:")
        logger.info(f"  最小值: {sorted_rl_steps[0]}")
        logger.info(f"  最大值: {sorted_rl_steps[-1]}")
        logger.info(f"  平均值: {statistics.mean(sorted_rl_steps):.1f}")
        logger.info(f"  P90: {sorted_rl_steps[int(n * 0.90)]}")
        logger.info(f"  P95: {sorted_rl_steps[int(n * 0.95)]}")
        logger.info(f"  P99: {sorted_rl_steps[int(n * 0.99)]}")

        # 关键：超过 max_steps=20 的会话
        over_20_steps = sum(1 for c in rl_steps_counts if c > 20)
        over_15_steps = sum(1 for c in rl_steps_counts if c > 15)
        logger.info(f"")
        logger.info(f"  ⚠️  超过 20 RL steps (会被截断): {over_20_steps} ({over_20_steps/len(rl_steps_counts)*100:.1f}%)")
        logger.info(f"  超过 15 RL steps: {over_15_steps} ({over_15_steps/len(rl_steps_counts)*100:.1f}%)")

    logger.info("=" * 60)

    return updated_playbooks


def main():
    parser = argparse.ArgumentParser(description='回填 effective_turn_count 字段')
    parser.add_argument('--input', required=True, help='输入 playbooks JSON 文件路径')
    parser.add_argument('--output', help='输出文件路径（默认：添加 _with_turns 后缀）')
    parser.add_argument('--inplace', action='store_true', help='原地覆盖输入文件')
    parser.add_argument('--raw-sessions', help='原始 session 数据文件路径（可选，用于精确计算）')

    args = parser.parse_args()

    # 加载 playbooks
    logger.info(f"Loading playbooks from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        playbooks = json.load(f)
    logger.info(f"Loaded {len(playbooks)} playbooks")

    # 加载原始 session 数据
    raw_session_map = {}
    if args.raw_sessions:
        raw_session_map = load_raw_sessions(args.raw_sessions)
    else:
        # 尝试从默认路径加载
        default_paths = [
            'data/raw_sessions.json',
            'outputs/raw_sessions.json',
            'data/sessions.json',
        ]
        for path in default_paths:
            if Path(path).exists():
                logger.info(f"Found raw sessions at {path}")
                raw_session_map = load_raw_sessions(path)
                break

        if not raw_session_map:
            logger.warning("No raw session data found. Turn counts may be incomplete.")
            logger.warning("Use --raw-sessions to specify the path to raw session data.")

    # 回填
    updated_playbooks = backfill_playbooks(playbooks, raw_session_map)

    # 确定输出路径
    if args.inplace:
        output_path = args.input
    elif args.output:
        output_path = args.output
    else:
        output_path = args.input.replace('.json', '_with_turns.json')

    # 保存
    logger.info(f"Saving to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_playbooks, f, ensure_ascii=False, indent=2)
    logger.info("Done!")


if __name__ == '__main__':
    main()