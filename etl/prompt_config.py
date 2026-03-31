# Copyright 2025 Nanyang Technological University (NTU), Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
# Priority Waterfall Decision Rules (STRICT - DO NOT MODIFY)
# =============================================================================

PRIORITY_WATERFALL_RULES = """
【🔴 动作决策全局瀑布流准则（严格遵守） 🔴】
作为客服，你每轮只能输出唯一的动作ID。面对买家输入，你必须按以下优先级从上到下匹配，一旦命中立即输出，禁止降级：

- [优先级 1 - 核心业务]：若买家表达了具体的业务诉求（如问参数、查物流、退换货等），必须直接选择对应的业务动作（如 pre_answer_spec, log_query_status）。系统底层会自动为你对用户补齐问候或安抚话术，你只需输出业务 Action。
- [优先级 2 - 信息收集]：若买家有业务诉求倾向，但意图模糊或缺失关键信息（如只发了一个链接），才能降级选择 gen_clarify 或 gen_verify_order。
- [优先级 3 - 纯社交兜底]：**只有当**买家输入纯问候（如"你好"、"在吗"）或纯发泄情绪，且**完全没有包含任何业务诉求时**，才可降级选择 gen_greet 或 gen_empathize。
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
                trigger_short = trigger[:120] + "..." if len(trigger) > 120 else trigger
                avoid_short = avoid[:100] + "..." if len(avoid) > 100 else avoid
                mistakes_lines.append(f"    ⚠️ 误触发场景: {trigger_short}\n    ✓ 正确做法: {avoid_short}")
        if mistakes_lines:
            mistakes_formatted = "\n  " + "\n  ".join(mistakes_lines)

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


def get_all_skill_ids() -> List[str]:
    """Get all skill IDs from SKILL_DICT."""
    return list(SKILL_DICT.keys())


# =============================================================================
# System Prompt Base (Dynamic - uses rich skill format)
# =============================================================================

SYSTEM_PROMPT_BASE = """你是一名专业的电商客服智能助手，擅长处理售前咨询、物流查询和售后问题。
你的目标是高效解决买家问题，促成交易或妥善处理售后，同时避免激怒买家。

{priority_waterfall_rules}

你需要在每一步选择正确的服务动作。你的输出格式必须是：

<tool_call>
[你的分析过程：买家想要什么？当前对话处于什么阶段？哪个动作最合适？]
<action>skill_id</action>

当前节点可用的动作及避坑指南如下：
{available_skills_rich}
"""


def get_system_prompt(
    include_waterfall_rules: bool = True,
    available_skills: List[str] = None,
    max_mistakes_per_skill: int = 2
) -> str:
    """
    Get the system prompt with priority waterfall rules and rich skill list.

    Args:
        include_waterfall_rules: Whether to include decision priority rules
        available_skills: List of available skill IDs. If None, uses all skills.
        max_mistakes_per_skill: Max mistakes to show per skill

    Returns:
        Formatted system prompt string
    """
    # Use all skills if not specified
    if available_skills is None:
        available_skills = get_all_skill_ids()

    # Format rich skill list
    available_skills_rich = format_available_skills_rich(available_skills, max_mistakes_per_skill)

    # Format waterfall rules
    waterfall = PRIORITY_WATERFALL_RULES if include_waterfall_rules else ""

    return SYSTEM_PROMPT_BASE.format(
        priority_waterfall_rules=waterfall,
        available_skills_rich=available_skills_rich
    )


# =============================================================================
# Backward Compatibility - Simple system prompt for parquet generation
# =============================================================================

def get_system_prompt_for_parquet(include_waterfall_rules: bool = True) -> str:
    """
    Get system prompt for parquet data generation.

    Uses ALL 31 skills since available_skills is determined at runtime by environment.
    The environment will validate against actual available_skills during training.

    Args:
        include_waterfall_rules: Whether to include decision priority rules

    Returns:
        Formatted system prompt string with all skills
    """
    return get_system_prompt(
        include_waterfall_rules=include_waterfall_rules,
        available_skills=get_all_skill_ids(),
        max_mistakes_per_skill=2
    )