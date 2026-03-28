"""Pydantic 数据模型"""
from typing import Optional
from pydantic import BaseModel


class BusinessOutcome(BaseModel):
    has_order: bool
    order_amount: float


class Node(BaseModel):
    buyer_text: str
    sentiment: str
    slot_updates: dict
    available_skills: list[str]
    transitions: dict[str, str]
    default_fallback: str


class Playbook(BaseModel):
    playbook_id: str
    session_id: str
    scenario: str
    subtype: str
    business_outcome: BusinessOutcome
    initial_slots: dict
    nodes: dict[str, Node]
    # 新增字段（可选，兼容旧数据）
    effective_turn_count: Optional[int] = None  # 总 turns 数（User + Agent）
    rl_steps: Optional[int] = None  # Agent action 数 = User turns 数


class Message(BaseModel):
    sent_by: str
    format: str
    content: str
    sent_at: str


class Session(BaseModel):
    session_id: str
    message_count: int
    has_order: bool
    order_amount: float
    messages: list[Message]


class TurnStats(BaseModel):
    """轮次统计信息"""
    total: int
    with_turn_info: int  # 有 rl_steps 字段的 playbook 数量
    min_turns: int
    max_turns: int
    avg_turns: float
    min_steps: int
    max_steps: int
    avg_steps: float
    p90_steps: int
    p95_steps: int
    p99_steps: int
    over_20_steps: int  # 超过 max_steps=20 的数量
    steps_distribution: dict[int, int]  # steps -> count