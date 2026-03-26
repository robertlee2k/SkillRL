#!/usr/bin/env python3
"""
Parallel ETL Pipeline for Customer Service Playbook Generation.

Uses multiprocessing to parallelize LLM API calls for faster processing.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import ETL modules
sys.path.insert(0, str(Path(__file__).parent))
from etl.batch import load_sessions, process_batch
from etl.classifier import classify_scene
from etl.pipeline import build_playbook, load_checkpoint, save_incremental, SAVE_INTERVAL
from etl.validator import validate_playbook, ValidationError


def process_single_session(session: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Process a single session: classify + build playbook + validate.

    Args:
        session: Cleaned session dict

    Returns:
        Tuple of (playbook or None, session_id)
    """
    session_id = session.get('session_id', 'unknown')

    try:
        # Stage 1: Classify scene
        scenario = classify_scene(session['turns'])
        session['scenario'] = scenario

        # Stage 2: Build playbook
        playbook = build_playbook(session)

        if playbook is None:
            return None, session_id

        # Stage 3: Validate
        try:
            validate_playbook(playbook)
            return playbook, session_id
        except ValidationError:
            return None, session_id

    except Exception as e:
        return None, session_id


def run_parallel_pipeline(
    input_file: str,
    output_file: str,
    min_turns: int = 2,
    num_workers: int = 10,
    resume: bool = True
) -> Dict[str, int]:
    """
    Run ETL pipeline with parallel processing.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        min_turns: Minimum turns required
        num_workers: Number of parallel workers
        resume: Whether to resume from checkpoint

    Returns:
        Stats dict
    """
    logger.info(f"Starting Parallel ETL Pipeline with {num_workers} workers")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")

    start_time = time.time()

    # Load checkpoint if resuming
    playbooks, processed_ids = ([], set())
    if resume:
        playbooks, processed_ids = load_checkpoint(output_file)
        logger.info(f"Loaded {len(playbooks)} playbooks from checkpoint")

    # Load and clean sessions
    sessions = load_sessions(input_file)
    logger.info(f"Loaded {len(sessions)} sessions")

    result = process_batch(sessions, min_turns=min_turns)
    cleaned = result['playbooks']
    logger.info(f"Cleaned {len(cleaned)} valid sessions")

    # Filter out already processed
    to_process = [s for s in cleaned if s.get('session_id') not in processed_ids]
    logger.info(f"To process: {len(to_process)} sessions ({len(cleaned) - len(to_process)} already done)")

    if not to_process:
        logger.info("Nothing to process. Done!")
        return {'total': len(cleaned), 'valid': len(playbooks), 'invalid': 0}

    # Process in parallel
    stats = {'total': len(cleaned), 'valid': len(playbooks), 'invalid': result['stats']['invalid']}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_session, s): s for s in to_process}

        completed = 0
        for future in as_completed(futures):
            playbook, session_id = future.result()

            if playbook:
                playbooks.append(playbook)
                processed_ids.add(session_id)
                stats['valid'] += 1
            else:
                stats['invalid'] += 1

            completed += 1

            # Log progress every 50
            if completed % 50 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(to_process) - completed) / rate if rate > 0 else 0
                logger.info(f"Progress: {completed}/{len(to_process)} ({completed/len(to_process)*100:.1f}%) - "
                           f"Rate: {rate:.2f}/s - ETA: {eta/60:.1f}min")

                # Save checkpoint
                save_incremental(playbooks, output_file)

    # Final save
    save_incremental(playbooks, output_file)

    elapsed = time.time() - start_time
    logger.info(f"\n=== 完成 ===")
    logger.info(f"耗时: {elapsed/60:.1f} 分钟")
    logger.info(f"成功: {stats['valid']}")
    logger.info(f"失败: {stats['invalid']}")
    logger.info(f"成功率: {stats['valid']/stats['total']*100:.1f}%")
    logger.info(f"速度: {len(to_process)/elapsed:.2f} 条/秒")

    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Parallel ETL Pipeline')
    parser.add_argument('--input', '-i', default='session_order_converted.json', help='Input file')
    parser.add_argument('--output', '-o', default='outputs/playbooks_full.json', help='Output file')
    parser.add_argument('--workers', '-w', type=int, default=10, help='Number of parallel workers')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh (ignore checkpoint)')

    args = parser.parse_args()

    run_parallel_pipeline(
        input_file=args.input,
        output_file=args.output,
        num_workers=args.workers,
        resume=not args.no_resume
    )