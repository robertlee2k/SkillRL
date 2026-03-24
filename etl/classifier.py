"""
Scene classification for customer service conversations.
Uses LLM-based classification for accuracy.
"""
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def classify_scene(turns: List[Dict[str, Any]]) -> str:
    """
    Classify scene type based on conversation content using LLM.

    Args:
        turns: List of turn dicts with 'role' and 'text' keys

    Returns:
        'presale', 'logistics', 'aftersale', or 'unknown'
    """
    # Handle empty input
    if not turns:
        logger.warning("[Classifier] Empty turns list, returning unknown")
        return 'unknown'

    # Extract first 3 turns of User text for classification
    user_texts = []
    for turn in turns[:6]:  # Check first 6 turns to get 3 User turns
        if turn.get('role') == 'User':
            text = turn.get('text', '') or turn.get('buyer_text', '') or ''
            if text.strip():
                user_texts.append(text.strip())
        if len(user_texts) >= 3:
            break

    if not user_texts:
        logger.warning("[Classifier] No User text found, returning unknown")
        return 'unknown'

    # Combine texts for LLM classification
    conversation_text = "\n".join(f"买家: {text}" for text in user_texts)

    # Call LLM for classification
    try:
        from .llm_generator import call_llm_for_classification
        scene = call_llm_for_classification(conversation_text)

        # Map 'trash' to 'unknown' for consistency
        if scene == 'trash':
            return 'unknown'
        return scene

    except Exception as e:
        logger.error(f"[Classifier] LLM classification failed: {e}")
        return 'unknown'