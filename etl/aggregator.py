# etl/aggregator.py
"""Turn aggregation core algorithm for ETL pipeline.

Implements aggregate_turns per design spec Section 3.3.
Converts multi-role messages into User/Agent strictly alternating turns.
"""

from typing import Dict, List, Any

from etl.parser import parse_system_message


# Design spec Section 3.1: Role mapping
ROLE_MAPPING: Dict[str, str] = {
    # 保留角色
    "BUYER": "User",

    # 合并为 Agent
    "ASSISTANT": "Agent",
    "QA": "Agent",
    "QA_VENDOR": "Agent",

    # MARKETING is handled by deletion (mapped to None)
    # SYSTEM is handled separately as slot_update
}


def extract_text(msg: Dict[str, Any]) -> str:
    """
    Extract text content from message.

    Args:
        msg: Message dict with 'content' field

    Returns:
        Text content string, empty string if not found
    """
    content = msg.get('content', '')
    if not content:
        return ''
    # Handle if content is already a string
    if isinstance(content, str):
        return content.strip()
    # Handle if content is a dict (e.g., {'text': '...'})
    if isinstance(content, dict):
        return content.get('text', '').strip()
    return str(content).strip()


def aggregate_turns(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    将多角色消息聚合为 User/Agent 严格交替的 Turns

    Args:
        messages: List of message dicts with 'sent_by' and 'content' fields

    Returns:
        Dict with:
            "turns": List of turn dicts with 'role', 'text', 'slot_updates'
            "initial_slots": Dict of session-level initial slot values

    Raises:
        ValueError: If turns don't alternate properly (consecutive same roles)
    """
    turns: List[Dict[str, Any]] = []
    current_role: str = None
    current_texts: List[str] = []
    current_slot_updates: Dict[str, bool] = {}  # Track slot updates for pending turn

    # 关键修复：全局初始槽位，处理SYSTEM消息置顶的Edge Case
    initial_slots: Dict[str, bool] = {}

    for msg in messages:
        # 跳过 MARKETING
        if msg.get('sent_by') == 'MARKETING':
            continue

        # 处理 SYSTEM
        if msg.get('sent_by') == 'SYSTEM':
            slot_update = parse_system_message(msg)
            if slot_update:
                # 关键修复：需要考虑 pending turn 的情况
                # 如果当前正在处理 Agent turn，则附加到 pending turn 的 slot_updates
                if current_role == 'Agent':
                    # Agent turn is pending, attach to it
                    current_slot_updates.update(slot_update)
                elif not turns:
                    # 会话最开始，存入全局initial_slots
                    initial_slots.update(slot_update)
                elif turns[-1]['role'] == 'Agent':
                    # 追加到最近的 Agent turn 的 metadata (已 flush 的)
                    turns[-1]['slot_updates'].update(slot_update)
                else:
                    # 前一个是User，存入全局initial_slots
                    initial_slots.update(slot_update)
            continue

        # 角色映射
        mapped_role = ROLE_MAPPING.get(msg.get('sent_by'))
        if mapped_role is None:
            # Unknown role or MARKETING (already handled)
            continue

        # 文本提取
        text = extract_text(msg)
        if not text:
            continue

        # 聚合逻辑
        if current_role == mapped_role:
            # 同角色连续，拼接文本
            current_texts.append(text)
        else:
            # 角色切换，保存上一轮
            if current_role:
                turns.append({
                    'role': current_role,
                    'text': ' '.join(current_texts),
                    'slot_updates': current_slot_updates.copy()
                })
            current_role = mapped_role
            current_texts = [text]
            current_slot_updates = {}  # Reset for new turn

    # 保存最后一轮
    if current_role:
        turns.append({
            'role': current_role,
            'text': ' '.join(current_texts),
            'slot_updates': current_slot_updates.copy()
        })

    # 校验二人转
    for i in range(len(turns) - 1):
        if turns[i]['role'] == turns[i+1]['role']:
            raise ValueError(f"角色交替错误: 连续两个 {turns[i]['role']}")

    return {
        "turns": turns,
        "initial_slots": initial_slots
    }