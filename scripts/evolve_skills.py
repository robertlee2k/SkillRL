#!/usr/bin/env python
"""
Offline Skill Evolution Script with MapReduce Architecture.

This script implements a "Data Flywheel" architecture with MapReduce:
1. Load failed trajectories from analyze_bad_cases.py output
2. Group failures by error action
3. Map Phase: Process failure chunks in parallel with LLM
4. Reduce Phase: Deduplicate and synthesize final mistakes
5. Update the skill registry JSON file

Usage:
    python scripts/evolve_skills.py \
        --trace_path outputs/recovered_bad_cases_step90.json \
        --output_path memory_data/customer_service/claude_style_skills.json \
        --min_failures 3 \
        --batch_size 10 \
        --max_workers 5

Environment Variables:
    VOLC_API_KEY - Volcano Engine (Doubao) API key
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants - Must match customer_service_env.py and config.py
# =============================================================================

VALID_SKILLS = {
    # General skills
    'gen_greet', 'gen_empathize', 'gen_clarify', 'gen_verify_order',
    'gen_hold', 'gen_transfer', 'gen_apologize', 'gen_close',
    # Presale skills
    'pre_query_product', 'pre_check_stock', 'pre_compare', 'pre_recommend',
    'pre_answer_spec', 'pre_check_promo', 'pre_guide_purchase',
    # Logistics skills
    'log_query_status', 'log_query_detail', 'log_estimate_arrival',
    'log_modify_address', 'log_contact_courier', 'log_delay_notify', 'log_lost_claim',
    # Aftersale skills
    'aft_check_policy', 'aft_collect_evidence', 'aft_initiate_refund',
    'aft_initiate_return', 'aft_initiate_exchange', 'aft_schedule_pickup',
    'aft_track_progress', 'aft_compensate', 'aft_reject_explain'
}

# Skill categories for organization
SKILL_CATEGORIES = {
    'general': ['gen_greet', 'gen_empathize', 'gen_clarify', 'gen_verify_order',
                'gen_hold', 'gen_transfer', 'gen_apologize', 'gen_close'],
    'presale': ['pre_query_product', 'pre_check_stock', 'pre_compare', 'pre_recommend',
                'pre_answer_spec', 'pre_check_promo', 'pre_guide_purchase'],
    'logistics': ['log_query_status', 'log_query_detail', 'log_estimate_arrival',
                  'log_modify_address', 'log_contact_courier', 'log_delay_notify', 'log_lost_claim'],
    'aftersale': ['aft_check_policy', 'aft_collect_evidence', 'aft_initiate_refund',
                  'aft_initiate_return', 'aft_initiate_exchange', 'aft_schedule_pickup',
                  'aft_track_progress', 'aft_compensate', 'aft_reject_explain']
}

# Skill descriptions (brief, for context)
SKILL_DESCRIPTIONS = {
    # General
    'gen_greet': '问候买家，建立友好开场',
    'gen_empathize': '安抚情绪，表达理解和同情',
    'gen_clarify': '澄清意图，确认买家需求',
    'gen_verify_order': '请求订单信息，核实订单号',
    'gen_hold': '请求等待，告知需要时间处理',
    'gen_transfer': '转人工服务',
    'gen_apologize': '道歉，承认服务不足',
    'gen_close': '结束会话，礼貌道别',
    # Presale
    'pre_query_product': '查询商品信息',
    'pre_check_stock': '查询库存状态',
    'pre_compare': '对比多个商品',
    'pre_recommend': '推荐合适商品',
    'pre_answer_spec': '回答规格参数问题',
    'pre_check_promo': '查询优惠活动',
    'pre_guide_purchase': '引导下单流程',
    # Logistics
    'log_query_status': '查询物流状态',
    'log_query_detail': '查询物流详情',
    'log_estimate_arrival': '预估到达时间',
    'log_modify_address': '修改收货地址',
    'log_contact_courier': '联系快递员',
    'log_delay_notify': '延迟到货通知',
    'log_lost_claim': '丢件理赔处理',
    # Aftersale
    'aft_check_policy': '查询退换货政策',
    'aft_collect_evidence': '收集问题证据',
    'aft_initiate_refund': '发起退款',
    'aft_initiate_return': '发起退货',
    'aft_initiate_exchange': '发起换货',
    'aft_schedule_pickup': '安排取件',
    'aft_track_progress': '跟踪售后进度',
    'aft_compensate': '赔偿处理',
    'aft_reject_explain': '拒绝申请并说明原因'
}

# Default batch size for chunking
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_WORKERS = 5


# =============================================================================
# LLM Client (Doubao via Volcano Engine)
# =============================================================================

class DoubaoClient:
    """Client for Doubao LLM via Volcano Engine API."""

    def __init__(self, max_tokens: int = 2048):
        api_key = os.getenv('VOLC_API_KEY')
        if not api_key:
            raise ValueError(
                "[DoubaoClient] VOLC_API_KEY environment variable not set! "
                "Please export VOLC_API_KEY=your_api_key"
            )

        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=api_key,
        )
        # Doubao Seed 2.0 Pro model
        self.model = "doubao-seed-2-0-pro-260215"
        self.max_tokens = max_tokens

        logger.info(f"[DoubaoClient] Initialized with model: {self.model}")

    def analyze(self, prompt: str) -> str:
        """Send prompt to Doubao and get response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:
            logger.error(f"[DoubaoClient] API call failed: {e}")
            raise


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ErrorContext:
    """Context for a single error action within a trajectory."""
    playbook_id: str
    scenario: str
    step: int
    action: str
    reasoning: str
    system_message: str
    dialogue_before: List[Dict[str, str]]
    patience_before: int
    patience_after: int


@dataclass
class ActionFailureGroup:
    """Group of failures for the same action."""
    action: str
    count: int
    contexts: List[ErrorContext]


# =============================================================================
# Core Logic - Data Loading & Grouping
# =============================================================================

def load_trace_data(trace_path: str) -> Dict[str, Any]:
    """Load bad case trace JSON from analyze_bad_cases.py output."""
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    with open(trace_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"[Load] Loaded trace data from {trace_path}")
    logger.info(f"[Load] Total bad cases: {len(data.get('bad_cases', []))}")

    return data


def extract_error_contexts(bad_cases: List[Dict]) -> List[ErrorContext]:
    """
    Extract all error contexts from bad cases.

    Each error context includes:
    - The action that failed
    - Model's reasoning before the action
    - System error message
    - Dialogue history before the error
    """
    contexts = []

    for case in bad_cases:
        playbook_id = case.get('playbook_id', 'unknown')
        scenario = case.get('scenario', 'unknown')
        dialogue_history = case.get('dialogue_history', [])
        error_actions = case.get('error_actions', [])

        for error in error_actions:
            step = error.get('step', 0)
            action = error.get('action', 'unknown')
            reasoning = error.get('reasoning', '')
            system_message = error.get('system_message', '')
            patience_before = error.get('patience_before', 0)
            patience_after = error.get('patience_after', 0)

            # Get dialogue before this step (last 3 turns)
            dialogue_before = dialogue_history[-6:] if len(dialogue_history) >= 6 else dialogue_history

            ctx = ErrorContext(
                playbook_id=playbook_id,
                scenario=scenario,
                step=step,
                action=action,
                reasoning=reasoning,
                system_message=system_message,
                dialogue_before=dialogue_before,
                patience_before=patience_before,
                patience_after=patience_after
            )
            contexts.append(ctx)

    logger.info(f"[Extract] Extracted {len(contexts)} error contexts")
    return contexts


def group_failures_by_action(
    contexts: List[ErrorContext],
    min_failures: int = 3
) -> List[ActionFailureGroup]:
    """
    Group error contexts by action, filter by minimum failure count.
    """
    action_to_contexts = defaultdict(list)

    for ctx in contexts:
        if ctx.action in VALID_SKILLS:
            action_to_contexts[ctx.action].append(ctx)

    groups = []
    for action, ctx_list in action_to_contexts.items():
        if len(ctx_list) >= min_failures:
            groups.append(ActionFailureGroup(
                action=action,
                count=len(ctx_list),
                contexts=ctx_list
            ))

    # Sort by count descending
    groups.sort(key=lambda g: g.count, reverse=True)

    logger.info(f"[Group] Found {len(groups)} actions with >= {min_failures} failures")
    for g in groups:
        logger.info(f"  - {g.action}: {g.count} failures")

    return groups


# =============================================================================
# Map Phase - Chunk Processing
# =============================================================================

def chunk_contexts(contexts: List[ErrorContext], batch_size: int) -> List[List[ErrorContext]]:
    """Split contexts into chunks of specified batch size."""
    chunks = []
    for i in range(0, len(contexts), batch_size):
        chunks.append(contexts[i:i + batch_size])
    return chunks


def format_context_for_prompt(ctx: ErrorContext) -> str:
    """Format a single error context for the prompt."""
    dialogue_str = ""
    for turn in ctx.dialogue_before:
        role = turn.get('role', 'unknown')
        content = turn.get('content', '')[:100]  # Truncate
        dialogue_str += f"  [{role}] {content}\n"

    return f"""
Example (Scenario: {ctx.scenario}, Step {ctx.step}):
Dialogue before error:
{dialogue_str}
Model's reasoning: {ctx.reasoning[:200] if ctx.reasoning else '(no reasoning)'}
System feedback: {ctx.system_message[:150] if ctx.system_message else '(none)'}
---"""


def build_map_prompt(action: str, description: str, chunk: List[ErrorContext]) -> str:
    """Build the prompt for MAP phase (chunk analysis)."""
    examples_block = "\n".join([
        format_context_for_prompt(ctx)
        for ctx in chunk
    ])

    prompt = f"""你是一名资深的电商客服训练专家，负责分析客服AI模型的错误决策模式。

## 任务背景

我们有一个客服AI模型，它在训练中选择了一个**错误的动作**，导致对话失败。
你的任务是分析这个动作的错误触发模式，并生成"避坑指南"，帮助模型避免类似的错误。

## 问题动作

**动作ID**: `{action}`
**动作描述**: {description}
**本批次样本数**: {len(chunk)}

## 错误案例

以下是模型错误选择该动作的失败案例：

{examples_block}

## 分析要求

请仔细分析以上案例，回答：

1. **触发条件**: 在什么样的对话上下文或买家意图下，模型容易**错误地**选择 `{action}`？
   - 买家说了什么话容易误导模型？
   - 对话处于什么阶段时模型容易犯错？
   - 有什么"陷阱"特征？

2. **避免方法**: 模型应该怎么做才能避免这个错误？
   - 正确的决策逻辑是什么？
   - 应该选择什么动作替代？
   - 为什么在这个场景下 `{action}` 是业务偏航？

## 输出格式

请严格输出以下JSON格式，不要有其他文字：

```json
[
  {{
    "trigger_condition": "描述触发错误的具体场景（1-2句话）",
    "how_to_avoid": "描述正确的做法和原因（1-2句话）"
  }},
  {{
    "trigger_condition": "另一个触发场景",
    "how_to_avoid": "对应的避免方法"
  }}
]
```

请生成 1-3 条最关键的避坑指南。只输出JSON数组，不要有任何解释文字。
"""

    return prompt


def parse_mistakes_response(response: str) -> List[Dict[str, str]]:
    """Parse LLM response to extract mistakes JSON array."""
    try:
        # Find JSON array in response
        json_start = response.find('[')
        json_end = response.rfind(']') + 1

        if json_start == -1 or json_end == 0:
            logger.warning("[Parse] No JSON array found in response")
            return []

        json_str = response[json_start:json_end]
        mistakes = json.loads(json_str)

        # Validate structure
        valid_mistakes = []
        for m in mistakes:
            if 'trigger_condition' in m and 'how_to_avoid' in m:
                valid_mistakes.append({
                    'trigger_condition': m['trigger_condition'].strip(),
                    'how_to_avoid': m['how_to_avoid'].strip()
                })

        return valid_mistakes

    except json.JSONDecodeError as e:
        logger.error(f"[Parse] JSON decode error: {e}")
        return []


def map_analyze_chunk(
    client: DoubaoClient,
    action: str,
    description: str,
    chunk: List[ErrorContext],
    chunk_idx: int,
    total_chunks: int
) -> Tuple[int, List[Dict[str, str]]]:
    """
    Map phase: Analyze a single chunk of failure contexts.

    Returns:
        (chunk_idx, list of mistakes)
    """
    try:
        prompt = build_map_prompt(action, description, chunk)
        response = client.analyze(prompt)
        mistakes = parse_mistakes_response(response)

        logger.info(f"  [Map] Chunk {chunk_idx + 1}/{total_chunks}: extracted {len(mistakes)} mistakes")
        return (chunk_idx, mistakes)

    except Exception as e:
        logger.error(f"  [Map] Chunk {chunk_idx + 1}/{total_chunks} failed: {e}")
        return (chunk_idx, [])


# =============================================================================
# Reduce Phase - Deduplication & Synthesis
# =============================================================================

def build_reduce_prompt(action: str, description: str, all_mistakes: List[Dict[str, str]]) -> str:
    """Build the prompt for REDUCE phase (deduplication and synthesis)."""
    mistakes_json = json.dumps(all_mistakes, ensure_ascii=False, indent=2)

    prompt = f"""你是一个业务规则归纳专家。针对客服动作 `{action}`，我们从海量报错中提取了以下避坑指南碎片：

{mistakes_json}

其中包含大量重复或相似的规则。请你对它们进行全局去重、合并和提炼。

## 动作信息

**动作ID**: `{action}`
**动作描述**: {description}
**原始规则数量**: {len(all_mistakes)}

## 任务要求

1. **去重**: 合并意思相近的触发条件
2. **提炼**: 提取最致命、最具代表性的 3-5 条终极避坑指南
3. **精简**: 每条规则要简洁明了，直击要点

## 输出格式

请严格输出以下JSON格式，不要有其他文字：

```json
[
  {{
    "trigger_condition": "描述触发错误的具体场景（1-2句话）",
    "how_to_avoid": "描述正确的做法和原因（1-2句话）"
  }},
  {{
    "trigger_condition": "另一个触发场景",
    "how_to_avoid": "对应的避免方法"
  }}
]
```

请生成 3-5 条最关键的终极避坑指南。只输出JSON数组，不要有任何解释文字。
"""

    return prompt


def reduce_mistakes(
    client: DoubaoClient,
    action: str,
    description: str,
    all_mapped_mistakes: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Reduce phase: Deduplicate and synthesize all mapped mistakes.

    Args:
        client: LLM client
        action: The skill/action name
        description: Skill description
        all_mapped_mistakes: All mistakes from MAP phase

    Returns:
        Final list of deduplicated, synthesized mistakes (3-5 items)
    """
    if not all_mapped_mistakes:
        return []

    # If we have few mistakes already, no need to reduce
    if len(all_mapped_mistakes) <= 5:
        logger.info(f"  [Reduce] Only {len(all_mapped_mistakes)} mistakes, skip reduce")
        return all_mapped_mistakes

    logger.info(f"  [Reduce] Synthesizing {len(all_mapped_mistakes)} mistakes into 3-5 final rules...")

    try:
        prompt = build_reduce_prompt(action, description, all_mapped_mistakes)
        response = client.analyze(prompt)
        final_mistakes = parse_mistakes_response(response)

        logger.info(f"  [Reduce] Synthesized to {len(final_mistakes)} final mistakes")
        return final_mistakes

    except Exception as e:
        logger.error(f"  [Reduce] Failed: {e}")
        # Fallback: return first 5 mistakes
        return all_mapped_mistakes[:5]


# =============================================================================
# Registry Management
# =============================================================================

def load_or_create_skills_registry(skills_path: str) -> Dict[str, Any]:
    """
    Load existing skills registry or create a new one from VALID_SKILLS.
    """
    if os.path.exists(skills_path):
        with open(skills_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        logger.info(f"[Registry] Loaded existing registry from {skills_path}")
        return registry

    # Create new registry with skeleton
    logger.info(f"[Registry] Creating new registry at {skills_path}")

    # Build general skills list
    general_skills = []
    for i, skill_id in enumerate(SKILL_CATEGORIES['general'], 1):
        general_skills.append({
            'skill_id': f"gen_{i:03d}",
            'skill_name': skill_id,
            'title': SKILL_DESCRIPTIONS.get(skill_id, '通用技能'),
            'principle': f"在适当的时候使用 {skill_id} 来提升服务质量",
            'when_to_apply': "根据对话上下文判断",
            'common_mistakes': []
        })

    # Build task-specific skills by scenario
    task_specific_skills = {}

    # Presale
    task_specific_skills['presale'] = [
        {
            'skill_id': f"pre_{i:03d}",
            'skill_name': skill_id,
            'title': SKILL_DESCRIPTIONS.get(skill_id, '售前技能'),
            'principle': f"在售前咨询中使用 {skill_id} 来促进转化",
            'when_to_apply': "当买家咨询商品相关信息时",
            'common_mistakes': []
        }
        for i, skill_id in enumerate(SKILL_CATEGORIES['presale'], 1)
    ]

    # Logistics
    task_specific_skills['logistics'] = [
        {
            'skill_id': f"log_{i:03d}",
            'skill_name': skill_id,
            'title': SKILL_DESCRIPTIONS.get(skill_id, '物流技能'),
            'principle': f"在物流查询中使用 {skill_id} 来解决配送问题",
            'when_to_apply': "当买家查询物流或配送相关问题时",
            'common_mistakes': []
        }
        for i, skill_id in enumerate(SKILL_CATEGORIES['logistics'], 1)
    ]

    # Aftersale
    task_specific_skills['aftersale'] = [
        {
            'skill_id': f"aft_{i:03d}",
            'skill_name': skill_id,
            'title': SKILL_DESCRIPTIONS.get(skill_id, '售后技能'),
            'principle': f"在售后处理中使用 {skill_id} 来妥善解决问题",
            'when_to_apply': "当买家发起退换货、投诉或赔偿请求时",
            'common_mistakes': []
        }
        for i, skill_id in enumerate(SKILL_CATEGORIES['aftersale'], 1)
    ]

    registry = {
        'general_skills': general_skills,
        'task_specific_skills': task_specific_skills,
        'common_mistakes': [],
        'metadata': {
            'source': 'generated by evolve_skills.py',
            'created_at': datetime.now().isoformat(),
            'version': '2.0-mapreduce'
        }
    }

    return registry


def find_skill_in_registry(
    registry: Dict[str, Any],
    skill_id: str
) -> Tuple[Optional[Dict], str]:
    """
    Find a skill in the registry by skill_name (e.g., 'gen_greet').

    Returns:
        (skill_dict, location) where location is 'general' or 'task_specific/{category}'
    """
    # Check general skills
    for skill in registry.get('general_skills', []):
        if skill.get('skill_name') == skill_id:
            return skill, 'general'

    # Check task-specific skills
    for category, skills in registry.get('task_specific_skills', {}).items():
        for skill in skills:
            if skill.get('skill_name') == skill_id:
                return skill, f'task_specific/{category}'

    return None, ''


def update_registry_with_mistakes(
    registry: Dict[str, Any],
    action: str,
    final_mistakes: List[Dict[str, str]],
    max_mistakes_per_action: int = 5
) -> int:
    """
    Update the registry with final synthesized mistakes for an action.
    """
    skill, location = find_skill_in_registry(registry, action)

    if skill is None:
        logger.warning(f"[Update] Skill not found in registry: {action}")
        return 0

    # Replace existing mistakes with final synthesized ones (trimmed to max)
    skill['common_mistakes'] = final_mistakes[:max_mistakes_per_action]

    logger.info(f"[Update] Updated {action} with {len(skill['common_mistakes'])} mistakes (location: {location})")
    return len(skill['common_mistakes'])


def save_registry(registry: Dict[str, Any], output_path: str):
    """Save the updated registry to JSON file."""
    registry['metadata']['last_updated'] = datetime.now().isoformat()
    registry['metadata']['update_count'] = registry['metadata'].get('update_count', 0) + 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    logger.info(f"[Save] Saved registry to {output_path}")


# =============================================================================
# Main Entry Point - MapReduce Pipeline
# =============================================================================

def process_action_with_mapreduce(
    client: DoubaoClient,
    group: ActionFailureGroup,
    batch_size: int,
    max_workers: int,
    max_mistakes_per_action: int
) -> List[Dict[str, str]]:
    """
    Process a single action using MapReduce architecture.

    1. Split contexts into chunks
    2. Map: Parallel LLM analysis of each chunk
    3. Reduce: Deduplicate and synthesize final mistakes
    """
    action = group.action
    description = SKILL_DESCRIPTIONS.get(action, '未知动作')
    contexts = group.contexts

    # Split into chunks
    chunks = chunk_contexts(contexts, batch_size)
    total_chunks = len(chunks)

    logger.info(f"  [MapReduce] {action}: {len(contexts)} failures -> {total_chunks} chunks")

    # === MAP Phase: Parallel chunk analysis ===
    all_mapped_mistakes = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                map_analyze_chunk,
                client,
                action,
                description,
                chunk,
                idx,
                total_chunks
            ): idx
            for idx, chunk in enumerate(chunks)
        }

        for future in as_completed(futures):
            try:
                chunk_idx, mistakes = future.result()
                all_mapped_mistakes.extend(mistakes)
            except Exception as e:
                logger.error(f"  [Map] Chunk failed with exception: {e}")

    logger.info(f"  [Map] Total extracted: {len(all_mapped_mistakes)} raw mistakes")

    # === REDUCE Phase: Deduplicate and synthesize ===
    final_mistakes = reduce_mistakes(
        client,
        action,
        description,
        all_mapped_mistakes
    )

    # Trim to max
    final_mistakes = final_mistakes[:max_mistakes_per_action]

    return final_mistakes


def evolve_skills(
    trace_path: str,
    output_path: str,
    min_failures: int = 3,
    max_mistakes_per_action: int = 5,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    dry_run: bool = False
):
    """
    Main skill evolution pipeline with MapReduce architecture.

    Args:
        trace_path: Path to bad_cases JSON
        output_path: Path to claude_style_skills.json
        min_failures: Minimum failures to analyze an action
        max_mistakes_per_action: Max final mistakes per action (after reduce)
        batch_size: Number of contexts per chunk in MAP phase
        max_workers: Number of parallel workers in MAP phase
        dry_run: If True, don't call LLM or save file
    """
    logger.info("=" * 60)
    logger.info("SKILL EVOLUTION PIPELINE (MapReduce)")
    logger.info("=" * 60)

    # Step 1: Load trace data
    trace_data = load_trace_data(trace_path)
    bad_cases = trace_data.get('bad_cases', [])

    if not bad_cases:
        logger.warning("No bad cases found in trace data. Exiting.")
        return

    # Step 2: Extract error contexts
    contexts = extract_error_contexts(bad_cases)

    # Step 3: Group by action
    groups = group_failures_by_action(contexts, min_failures)

    if not groups:
        logger.info("No actions with enough failures to analyze. Exiting.")
        return

    # Step 4: Load or create registry
    registry = load_or_create_skills_registry(output_path)

    # Step 5: Process each action with MapReduce
    if dry_run:
        logger.info("[Dry Run] Skipping LLM analysis")
        return

    client = DoubaoClient()

    total_added = 0
    stats = {
        'actions_processed': 0,
        'actions_failed': 0,
        'total_mistakes': 0
    }

    for group in groups:
        logger.info(f"\n[Action] Processing: {group.action} ({group.count} failures)")

        try:
            final_mistakes = process_action_with_mapreduce(
                client=client,
                group=group,
                batch_size=batch_size,
                max_workers=max_workers,
                max_mistakes_per_action=max_mistakes_per_action
            )

            if final_mistakes:
                added = update_registry_with_mistakes(
                    registry,
                    group.action,
                    final_mistakes,
                    max_mistakes_per_action
                )
                total_added += added
                stats['total_mistakes'] += len(final_mistakes)

                logger.info(f"  [Result] {group.action}: {len(final_mistakes)} final mistakes")
                for m in final_mistakes:
                    logger.info(f"    - Trigger: {m['trigger_condition'][:50]}...")
                    logger.info(f"      Avoid: {m['how_to_avoid'][:50]}...")

            stats['actions_processed'] += 1

        except Exception as e:
            logger.error(f"  [Error] Failed to process {group.action}: {e}")
            traceback.print_exc()
            stats['actions_failed'] += 1
            continue

    # Step 6: Save updated registry
    save_registry(registry, output_path)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EVOLUTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total bad cases analyzed: {len(bad_cases)}")
    logger.info(f"Actions with >= {min_failures} failures: {len(groups)}")
    logger.info(f"Actions processed: {stats['actions_processed']}")
    logger.info(f"Actions failed: {stats['actions_failed']}")
    logger.info(f"Total final mistakes: {stats['total_mistakes']}")
    logger.info(f"Registry saved to: {output_path}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Evolve skills from failure trajectories with MapReduce'
    )
    parser.add_argument(
        '--trace_path',
        type=str,
        default='outputs/recovered_bad_cases_step90.json',
        help='Path to bad cases trace JSON'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='memory_data/customer_service/claude_style_skills.json',
        help='Path to output skills registry JSON'
    )
    parser.add_argument(
        '--min_failures',
        type=int,
        default=3,
        help='Minimum failures to analyze an action'
    )
    parser.add_argument(
        '--max_mistakes_per_action',
        type=int,
        default=5,
        help='Maximum final mistakes to keep per action'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Number of contexts per chunk in MAP phase'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help='Number of parallel workers in MAP phase'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Run without calling LLM or saving'
    )

    args = parser.parse_args()

    evolve_skills(
        trace_path=args.trace_path,
        output_path=args.output_path,
        min_failures=args.min_failures,
        max_mistakes_per_action=args.max_mistakes_per_action,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()