# etl/cleaner.py
"""Session cleaning pipeline for ETL.

Implements clean_session per design spec Section 4.
Validates and cleans sessions according to Red Line requirements.
"""

import logging
from typing import Dict, Any, Optional

from etl.aggregator import aggregate_turns

logger = logging.getLogger(__name__)


def validate_user_agent_alternation(turns: list) -> bool:
    """
    Validate strict User-Agent alternation.
    Red Line: Must strictly alternate between User and Agent.

    Args:
        turns: List of turn dicts with 'role' field

    Returns:
        True if alternation is valid, False otherwise
    """
    if not turns:
        return False

    # Must start with User
    if turns[0]['role'] != 'User':
        return False

    for i in range(1, len(turns)):
        if turns[i]['role'] == turns[i-1]['role']:
            return False

    return True


def clean_session(
    session: Dict[str, Any],
    min_turns: int = 2
) -> Optional[Dict[str, Any]]:
    """
    Clean a single session according to design spec Section 4.

    Handles sessions that start with Agent messages by removing leading
    Agent turns to preserve valuable user-agent interactions.

    Args:
        session: Session dict with 'messages' field
        min_turns: Minimum number of turns required (default 2)

    Returns:
        Cleaned session dict or None if session is invalid
    """
    messages = session.get('messages', [])

    # Aggregate turns
    aggregated = aggregate_turns(messages)
    turns = aggregated['turns']
    initial_slots = aggregated['initial_slots']

    # Handle sessions that start with Agent messages
    # These are typically:
    # 1. Agent sends product image/video first, then user asks questions
    # 2. Agent proactively reaches out (marketing or service)
    # We remove leading Agent turns to preserve the user-agent interaction
    if turns and turns[0]['role'] == 'Agent':
        # Find the first User turn
        first_user_idx = None
        for i, turn in enumerate(turns):
            if turn['role'] == 'User':
                first_user_idx = i
                break

        if first_user_idx is not None:
            # Remove leading Agent turns
            removed_count = first_user_idx
            turns = turns[first_user_idx:]
            logger.debug(f"Removed {removed_count} leading Agent turns from session {session.get('session_id')}")
        else:
            # All Agent messages, filter out
            logger.debug(f"Session {session.get('session_id')} has only Agent messages, filtered")
            return None

    # Validate minimum length
    if len(turns) < min_turns:
        return None

    # Validate User-Agent alternation
    if not validate_user_agent_alternation(turns):
        return None

    return {
        'session_id': session.get('session_id'),
        'has_order': session.get('has_order', False),
        'order_amount': float(session.get('order_amount', 0.0)),
        'turns': turns,
        'initial_slots': initial_slots
    }