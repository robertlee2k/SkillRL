# etl/classifier.py
from typing import List, Dict, Any
from .config import SCENE_PATTERNS

# Define priority order for tie-breaking
SCENE_PRIORITY = ['presale', 'logistics', 'aftersale']

def classify_scene(turns: List[Dict[str, Any]]) -> str:
    """
    Classify scene type based on conversation content.
    Returns: 'presale', 'logistics', 'aftersale', or 'unknown'
    """
    # Handle empty/None input
    if not turns:
        return 'unknown'

    # Combine all user content
    user_content = ' '.join(
        t.get('text', '') or '' for t in turns if t.get('role') == 'User'
    )

    if not user_content.strip():
        return 'unknown'

    # Count matches for each category
    scores = {}
    for category, patterns in SCENE_PATTERNS.items():
        score = sum(1 for p in patterns if p in user_content)
        scores[category] = score

    # Return highest scoring category with deterministic tie-breaking
    max_score = max(scores.values())
    if max_score == 0:
        return 'unknown'

    # Tie-breaking: return first category in priority order with max score
    for category in SCENE_PRIORITY:
        if scores.get(category, 0) == max_score:
            return category

    return 'unknown'