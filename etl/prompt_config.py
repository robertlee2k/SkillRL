# Copyright 2025 Nanyang Technological University (NTU), Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Shared Prompt Configuration for Customer Service Agent.

This module provides centralized prompt components that are reused by:
1. etl/rl_interfaces.py - Runtime prompt generation during training
2. scripts/prepare_cs_data.py - Offline training data preparation

This ensures DRY (Don't Repeat Yourself) principle and prevents desync.
"""

import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


# =============================================================================
# Skill Registry Loading
# =============================================================================

SKILL_REGISTRY_PATH = os.path.join(
    os.path.dirname(__file__),
    "../memory_data/customer_service/claude_style_skills.json"
)

SKILL_DICT: Dict[str, Dict[str, Any]] = {}

def load_skill_registry() -> Dict[str, Dict[str, Any]]:
    """Load skill registry from JSON file and build lookup dict by skill_name."""
    global SKILL_DICT
    if not os.path.exists(SKILL_REGISTRY_PATH):
        logger.warning(f"[PromptConfig] Skill registry not found: {SKILL_REGISTRY_PATH}")
        return {}

    try:
        with open(SKILL_REGISTRY_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        skill_lookup = {}
        for skill in data.get('general_skills', []):
            skill_name = skill.get('skill_name', '')
            if skill_name:
                skill_lookup[skill_name] = skill

        for scenario, skills in data.get('task_specific_skills', {}).items():
            for skill in skills:
                skill_name = skill.get('skill_name', '')
                if skill_name:
                    skill_lookup[skill_name] = skill

        SKILL_DICT = skill_lookup
        logger.info(f"[PromptConfig] Loaded {len(SKILL_DICT)} skills from registry")
        return skill_lookup

    except Exception as e:
        logger.error(f"[PromptConfig] Failed to load skill registry: {e}")
        return {}


# Load at module import
SKILL_DICT = load_skill_registry()


# =============================================================================
# Priority Waterfall Decision Rules
# =============================================================================

PRIORITY_WATERFALL_RULES = """
### 决策优先级规则（必须严格遵守）

按以下优先级顺序选择动作：

1. **场景匹配优先**: 首先识别当前场景类型（售前/物流/售后），优先选择该场景的专用动作
2. **用户诉求响应优先**: 有明确诉求时，优先响应用户核心问题，避免冗余澄清或安抚
3. **信息完整性优先**: 信息不全时，优先收集必要信息（订单号、商品信息等），而非跳过处理环节
4. **自助解答优先**: 常见问题优先自助解答（知识库查询、规格参数等），转人工仅作为最后兜底
5. **避免动作滥用**:
   - 禁止无明确负面情绪时触发 gen_empathize
   - 禁止需求已明确时触发 gen_clarify
   - 禁止信息完整时重复触发 gen_verify_order
   - 禁止常见问题可解答时直接转人工 gen_transfer
"""


# =============================================================================
# Rich Skill Formatting
# =============================================================================

def format_skill_with_mistakes(skill_name: str, skill_info: Dict[str, Any], max_mistakes: int = 2) -> str:
    """
    Format a skill with rich information including common mistakes.

    Args:
        skill_name: The skill ID (e.g., 'gen_clarify')
        skill_info: Skill info from SKILL_DICT containing title, principle, common_mistakes
        max_mistakes: Maximum number of mistakes to include (default 2)

    Returns:
        Formatted skill description string
    """
    if not skill_info:
        return f"**{skill_name}**"

    title = skill_info.get('title', skill_name)
    principle = skill_info.get('principle', '')

    mistakes = skill_info.get('common_mistakes', [])
    mistakes_formatted = ""
    if mistakes:
        mistakes_lines = []
        for m in mistakes[:max_mistakes]:
            trigger = m.get('trigger_condition', '')
            avoid = m.get('how_to_avoid', '')
            if trigger and avoid:
                # Truncate for readability
                trigger_short = trigger[:100] + "..." if len(trigger) > 100 else trigger
                avoid_short = avoid[:80] + "..." if len(avoid) > 80 else avoid
                mistakes_lines.append(f"    - ⚠️ {trigger_short}\n      → {avoid_short}")
        if mistakes_lines:
            mistakes_formatted = "\n  【常见误用场景】:\n" + "\n".join(mistakes_lines)

    return f"**{skill_name}**: {title}\n  用途: {principle}{mistakes_formatted}"


def format_available_skills_rich(available_skills: List[str], max_mistakes_per_skill: int = 2) -> str:
    """
    Format available skills list with rich information from SKILL_DICT.

    Args:
        available_skills: List of skill IDs available at current node
        max_mistakes_per_skill: Max mistakes to show per skill

    Returns:
        Formatted skills section string
    """
    formatted_lines = []
    for skill in available_skills:
        skill_info = SKILL_DICT.get(skill, {})
        formatted_lines.append(format_skill_with_mistakes(skill, skill_info, max_mistakes_per_skill))

    return "\n\n".join(formatted_lines)


# =============================================================================
# System Prompt Base
# =============================================================================

SYSTEM_PROMPT_BASE = """你是一名专业的电商客服智能助手，擅长处理售前咨询、物流查询和售后问题。
你的目标是高效解决买家问题，促成交易或妥善处理售后，同时避免激怒买家。
{priority_waterfall_rules}
你需要在每一步选择正确的服务动作。你的输出格式必须是：

<tool_call>
[你的分析过程：买家想要什么？当前对话处于什么阶段？哪个动作最合适？]
<action>skill_id</action>

可用的动作 ID 包括：
- 通用: gen_greet, gen_empathize, gen_clarify, gen_verify_order, gen_hold, gen_transfer, gen_apologize, gen_close
- 售前: pre_query_product, pre_check_stock, pre_compare, pre_recommend, pre_answer_spec, pre_check_promo, pre_guide_purchase
- 物流: log_query_status, log_query_detail, log_estimate_arrival, log_modify_address, log_contact_courier, log_delay_notify, log_lost_claim
- 售后: aft_check_policy, aft_collect_evidence, aft_initiate_refund, aft_initiate_return, aft_initiate_exchange, aft_schedule_pickup, aft_track_progress, aft_compensate, aft_reject_explain
"""


def get_system_prompt(include_waterfall_rules: bool = True) -> str:
    """
    Get the system prompt with optional priority waterfall rules.

    Args:
        include_waterfall_rules: Whether to include decision priority rules

    Returns:
        Formatted system prompt string
    """
    if include_waterfall_rules:
        return SYSTEM_PROMPT_BASE.format(priority_waterfall_rules=PRIORITY_WATERFALL_RULES)
    else:
        return SYSTEM_PROMPT_BASE.format(priority_waterfall_rules="")