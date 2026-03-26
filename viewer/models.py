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