# etl/batch.py
"""Batch processing utilities for ETL pipeline."""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from .cleaner import clean_session

logger = logging.getLogger(__name__)


def load_sessions(file_path: str) -> List[Dict[str, Any]]:
    """Load sessions from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Session file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in session file: {e}")


def save_playbooks(playbooks: List[Dict[str, Any]], file_path: str) -> None:
    """Save playbooks to JSON file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(playbooks, f, ensure_ascii=False, indent=2)


def process_batch(
    sessions: List[Dict[str, Any]],
    min_turns: int = 2
) -> Dict[str, Any]:
    """
    Process a batch of sessions.
    Returns dict with 'playbooks' and 'stats'.
    """
    results = []
    stats = {'total': len(sessions), 'valid': 0, 'invalid': 0}

    for session in sessions:
        cleaned = clean_session(session, min_turns=min_turns)
        if cleaned:
            results.append(cleaned)
            stats['valid'] += 1
        else:
            stats['invalid'] += 1

    logger.info(f"Processed {stats['total']} sessions: {stats['valid']} valid, {stats['invalid']} invalid")
    return {'playbooks': results, 'stats': stats}