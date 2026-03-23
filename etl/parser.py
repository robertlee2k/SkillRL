# etl/parser.py
"""Message parsing functions for ETL pipeline.

Handles parsing of SYSTEM messages and extracting slot updates
from conversation data.
"""

import json
import re
from typing import Dict, Any, Optional


# Design spec Section 3.1: System message patterns for slot updates
SYSTEM_SLOT_PATTERNS: Dict[str, Optional[Dict[str, bool]]] = {
    # 业务关键节点 → 槽位更新
    "买家已发起退款": {"refund_initiated": True},
    "订单已签收": {"order_signed": True},
    "买家已付款": {"order_paid": True},

    # 无意义提示 → 直接删除 (None means drop the message)
    "买家已读": None,
    "消息已送达": None,
}


def parse_system_message(msg: Dict[str, Any]) -> Dict[str, bool]:
    """
    Parse SYSTEM message to extract slot updates.

    Follows design spec Section 3.1.

    Args:
        msg: Message dict with 'sent_by' and 'content' fields

    Returns:
        Dict of extracted slot updates (key-value pairs)
    """
    content = msg.get('content', '')
    if not content:
        return {}

    # First check Chinese pattern matching (Section 3.1)
    for pattern, slots in SYSTEM_SLOT_PATTERNS.items():
        if pattern in content:
            if slots is None:
                # None means drop/ignore this message
                return {}
            return slots.copy()

    try:
        # Try to parse as JSON
        data = json.loads(content)
        if isinstance(data, dict) and 'slot_update' in data:
            return data['slot_update']
    except json.JSONDecodeError:
        pass

    # Try to extract from text patterns
    return extract_slot_updates(content)


def extract_slot_updates(text: str) -> Dict[str, bool]:
    """
    Extract slot updates from message text using patterns.

    Patterns supported:
    - Order ID: ORD-XXXXX or ord-XXXXX (case insensitive)
      Sets order_id_collected: true
    - Product ID: PROD-XXXXX or prod-XXXXX (case insensitive)
      Sets product_id_collected: true

    Args:
        text: Message text to extract from

    Returns:
        Dict of extracted slots with boolean values
    """
    slots: Dict[str, bool] = {}

    # Order ID pattern (e.g., ORD-12345, ord-12345)
    # Sets order_id_collected to True when found
    order_match = re.search(r'[Oo][Rr][Dd]-?\d+', text)
    if order_match:
        slots['order_id_collected'] = True

    # Product ID pattern (e.g., PROD-67890, prod-67890)
    # Sets product_id_collected to True when found
    product_match = re.search(r'[Pp][Rr][Oo][Dd]-?\d+', text)
    if product_match:
        slots['product_id_collected'] = True

    return slots