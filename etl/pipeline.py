# etl/pipeline.py
"""Full ETL pipeline orchestration.

Implements run_pipeline for complete end-to-end processing.
Uses LLM for scene classification and playbook generation.
Supports incremental saving and checkpoint recovery.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .batch import load_sessions, process_batch
from .classifier import classify_scene
from .validator import validate_playbook, ValidationError

logger = logging.getLogger(__name__)

# 每 N 条保存一次
SAVE_INTERVAL = 50


def load_checkpoint(output_file: str) -> tuple:
    """
    Load existing playbooks for checkpoint recovery.

    Returns:
        Tuple of (playbooks list, processed_session_ids set)
    """
    if Path(output_file).exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                playbooks = json.load(f)
            processed_ids = {pb['session_id'] for pb in playbooks}
            logger.info(f"[Checkpoint] Loaded {len(playbooks)} existing playbooks")
            return playbooks, processed_ids
        except Exception as e:
            logger.warning(f"[Checkpoint] Failed to load existing file: {e}")
    return [], set()


def save_incremental(playbooks: List[Dict], output_file: str) -> None:
    """Save playbooks incrementally (atomic write)."""
    temp_file = f"{output_file}.tmp"
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(playbooks, f, ensure_ascii=False, indent=2)
        # Atomic rename
        Path(temp_file).rename(output_file)
        logger.info(f"[Checkpoint] Saved {len(playbooks)} playbooks to {output_file}")
    except Exception as e:
        logger.error(f"[Checkpoint] Failed to save: {e}")
        if Path(temp_file).exists():
            Path(temp_file).unlink()


def run_pipeline(
    input_file: str,
    output_file: str,
    min_turns: int = 2,
    resume: bool = True
) -> Dict[str, int]:
    """
    Run complete ETL pipeline with checkpoint support.

    Stage 1: Load raw sessions
    Stage 2: Aggregate and clean
    Stage 3: Classify scenes (via LLM)
    Stage 4: Build and validate playbooks (via LLM)

    Args:
        input_file: Path to input JSON file with raw sessions
        output_file: Path to output JSON file for playbooks
        min_turns: Minimum number of turns required for valid session
        resume: Whether to resume from checkpoint (default True)

    Returns:
        Stats dict with total, valid, invalid counts
    """
    logger.info(f"Starting ETL pipeline: {input_file} -> {output_file}")

    # Load checkpoint if resuming
    playbooks, processed_ids = ([], set())
    if resume:
        playbooks, processed_ids = load_checkpoint(output_file)

    # Stage 1: Load
    sessions = load_sessions(input_file)
    logger.info(f"Loaded {len(sessions)} sessions")

    # Stage 2: Clean
    result = process_batch(sessions, min_turns=min_turns)
    cleaned = result['playbooks']
    # Start with batch stats (includes cleaning failures as invalid)
    stats = {'total': result['stats']['total'], 'valid': len(playbooks), 'invalid': result['stats']['invalid']}
    logger.info(f"Cleaned {len(cleaned)} valid sessions")

    # Count skipped due to checkpoint
    skipped = 0

    # Stage 3 & 4: Classify and build playbooks using LLM
    for i, session in enumerate(cleaned):
        session_id = session.get('session_id', f'unknown_{i}')

        # Skip if already processed (checkpoint recovery)
        if session_id in processed_ids:
            skipped += 1
            continue

        # Stage 3: Classify scene using LLM
        try:
            scenario = classify_scene(session['turns'])
            session['scenario'] = scenario
            logger.info(f"[{session_id}] Classified as: {scenario}")
        except Exception as e:
            logger.error(f"[{session_id}] Classification failed: {e}")
            stats['invalid'] += 1
            continue

        # Stage 4: Build playbook using LLM
        playbook = build_playbook(session)

        if playbook is None:
            logger.warning(f"[{session_id}] Playbook generation failed")
            stats['invalid'] += 1
            continue

        # Validate playbook
        try:
            validate_playbook(playbook)
            playbooks.append(playbook)
            stats['valid'] += 1
            processed_ids.add(session_id)
            logger.info(f"[{session_id}] Playbook generated successfully with {len(playbook['nodes'])} nodes")
        except ValidationError as e:
            logger.warning(f"[{session_id}] Playbook validation failed: {e}")
            stats['invalid'] += 1

        # Incremental save every SAVE_INTERVAL playbooks
        if len(playbooks) % SAVE_INTERVAL == 0:
            save_incremental(playbooks, output_file)

    # Final save
    save_incremental(playbooks, output_file)

    if skipped > 0:
        logger.info(f"Skipped {skipped} already processed sessions (checkpoint)")
    logger.info(f"Pipeline complete: {stats['valid']} valid, {stats['invalid']} invalid")

    return stats


def build_playbook(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build playbook JSON from cleaned session using LLM.

    Args:
        session: Cleaned session dict with 'session_id', 'turns', 'initial_slots', 'scenario'

    Returns:
        Playbook dict ready for validation, or None on failure
    """
    turns = session.get('turns', [])
    session_id = session.get('session_id', 'unknown')
    scenario = session.get('scenario', 'unknown')
    initial_slots = session.get('initial_slots', {})

    if not turns:
        logger.warning(f"[{session_id}] No turns to build playbook from")
        return None

    # 计算有效对话轮次（经过清洗后的 User-Agent 交替回合数）
    # 每个回合包含一个 User turn + 一个 Agent turn，所以 effective_turn_count = turns 总数 / 2
    # 但更准确的定义是：turns 列表中的条目数（每个条目代表一个角色的一轮发言）
    effective_turn_count = len(turns)

    # RL steps = Agent 需要执行的 action 数量 = User turns 数量
    # 因为每个 User turn 触发一次 Agent 响应
    # 如果 turns 以 User 开头且结尾（标准情况）：rl_steps = (effective_turn_count + 1) // 2
    # 实际上：rl_steps = count of User turns
    user_turn_count = sum(1 for t in turns if t['role'] == 'User')
    rl_steps = user_turn_count

    # Format conversation for LLM
    conversation_lines = []
    for turn in turns:
        role = turn.get('role', 'Unknown')
        text = turn.get('text', '')
        if role == 'User':
            conversation_lines.append(f"买家: {text}")
        elif role == 'Agent':
            conversation_lines.append(f"客服: {text}")

    conversation_text = "\n".join(conversation_lines)

    # Call LLM to generate playbook
    try:
        from .llm_generator import call_llm_for_playbook
        llm_result = call_llm_for_playbook(conversation_text, session_id)

        if llm_result is None:
            return None

        # Build complete playbook with metadata
        playbook = {
            'playbook_id': f"{scenario}_{session_id}",
            'session_id': session_id,  # 用于可视化溯源
            'scenario': scenario,
            'subtype': 'general',
            'effective_turn_count': effective_turn_count,  # 总 turns 数（User + Agent）
            'rl_steps': rl_steps,  # Agent 需执行的 action 数 = User turns 数
            'business_outcome': {
                'has_order': session.get('has_order', False),
                'order_amount': session.get('order_amount', 0.0)
            },
            'initial_slots': initial_slots,
            'nodes': llm_result.get('nodes', {})
        }

        return playbook

    except Exception as e:
        logger.error(f"[{session_id}] LLM playbook generation failed: {e}")
        return None