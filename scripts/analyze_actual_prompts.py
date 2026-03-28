#!/usr/bin/env python3
"""
真实 Prompt 长度分析脚本

模拟 veRL 训练时实际构建的 Prompt，包括：
1. 初始 System Prompt (来自 parquet 文件)
2. CustomerServicePromptBuilder 构建的状态提示
3. 对话历史累积

Usage:
    python scripts/analyze_actual_prompts.py
"""

import json
import os
from typing import Dict, List, Any
import pandas as pd

# 从 rl_interfaces.py 复制的模板
CUSTOMER_SERVICE_TEMPLATE_NO_HIS = """你是一名专业的电商客服智能助手，擅长处理售前咨询、物流查询和售后问题。

## 当前对话状态

**场景类型**: {scenario}
**当前节点**: {node_id}
**买家情绪**: {sentiment}

**买家最新消息**:
{buyer_text}

**当前槽位状态**:
{slots_formatted}

## 可用动作列表

以下是你当前可以选择的动作：

{available_skills_formatted}

## 动作说明

{skill_descriptions}

## 输出格式要求

你**必须**严格按照以下 XML 格式输出，先写思考过程，最后输出动作 ID：

<think>
在这里写下你的分析过程。
</think>

<action>这里填写对应的动作ID</action>

【标准输出示例】：
<think>
买家正在询问这款奶瓶是否带有把手，属于售前关于产品规格的咨询。此时应该选择回答规格问题的动作。
</think>

<action>pre_answer_spec</action>
"""

CUSTOMER_SERVICE_TEMPLATE = """你是一名专业的电商客服智能助手，擅长处理售前咨询、物流查询和售后问题。你的任务是: {task_description}

## 历史对话记录

在当前步骤之前，你已经执行了 {step_count} 步操作。以下是最近的 {history_length} 条对话和操作记录：

{action_history}

## 当前对话状态

**场景类型**: {scenario}
**当前节点**: {node_id}
**当前步骤**: {current_step}
**买家情绪**: {sentiment}

**买家最新消息**:
{buyer_text}

**当前槽位状态**:
{slots_formatted}

## 可用动作列表

以下是你当前可以选择的动作：

{available_skills_formatted}

## 动作说明

{skill_descriptions}

## 输出格式要求

你**必须**严格按照以下 XML 格式输出，先写思考过程，最后输出动作 ID：

<think>
在这里写下你的分析过程。
</think>

<action>这里填写对应的动作ID</action>

【标准输出示例】：
<think>
买家正在询问这款奶瓶是否带有把手，属于售前关于产品规格的咨询。此时应该选择回答规格问题的动作。
</think>

<action>pre_answer_spec</action>
"""

# 从 config.py 复制的 skill definitions
SKILL_DEFINITIONS = {
    'gen_greet': {'description': '问候'},
    'gen_empathize': {'description': '安抚情绪'},
    'gen_clarify': {'description': '澄清意图'},
    'gen_verify_order': {'description': '请求订单信息'},
    'gen_hold': {'description': '请求等待'},
    'gen_transfer': {'description': '转人工'},
    'gen_apologize': {'description': '致歉'},
    'gen_close': {'description': '结束会话'},
    'pre_query_product': {'description': '查询商品'},
    'pre_check_stock': {'description': '查库存'},
    'pre_compare': {'description': '商品对比'},
    'pre_recommend': {'description': '推荐'},
    'pre_answer_spec': {'description': '回答规格问题'},
    'pre_check_promo': {'description': '查优惠'},
    'pre_guide_purchase': {'description': '引导下单'},
    'log_query_status': {'description': '查物流状态'},
    'log_query_detail': {'description': '查物流详情'},
    'log_estimate_arrival': {'description': '预计到达时间'},
    'log_modify_address': {'description': '修改地址'},
    'log_contact_courier': {'description': '联系快递员'},
    'log_delay_notify': {'description': '延迟通知'},
    'log_lost_claim': {'description': '丢件理赔'},
    'aft_check_policy': {'description': '查退换政策'},
    'aft_collect_evidence': {'description': '收集证据'},
    'aft_initiate_refund': {'description': '发起退款'},
    'aft_initiate_return': {'description': '发起退货'},
    'aft_initiate_exchange': {'description': '发起换货'},
    'aft_schedule_pickup': {'description': '安排取件'},
    'aft_track_progress': {'description': '跟踪进度'},
    'aft_compensate': {'description': '赔偿'},
    'aft_reject_explain': {'description': '拒绝说明'},
}

VALID_SKILLS = list(SKILL_DEFINITIONS.keys())


def estimate_tokens(char_count: int) -> int:
    """
    粗略估算 token 数量

    中文: 约 1.5 字符/token
    英文/数字/符号: 约 4 字符/token
    平均估算: 约 2 字符/token
    """
    return char_count // 2


def load_playbooks(path: str) -> List[Dict[str, Any]]:
    """加载 playbook 数据"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_skills_with_desc(skills: List[str]) -> str:
    """格式化可用技能列表"""
    lines = []
    for skill in skills:
        desc = SKILL_DEFINITIONS.get(skill, {}).get('description', '未知动作')
        lines.append(f"- `{skill}`: {desc}")
    return '\n'.join(lines)


def simulate_dialogue_history(playbook: Dict[str, Any], max_steps: int = 20) -> List[str]:
    """
    模拟对话历史累积

    返回每一步可能的对话历史文本长度
    """
    nodes = playbook.get('nodes', {})
    history_lengths = []

    # 模拟每一步的对话历史
    # 假设每步增加: "买家: {buyer_text}" (~100-500 chars) + "客服: [Action: skill_id]" (~50 chars)
    dialogue_chars = 0

    for step in range(max_steps):
        # 模拟对话历史增长
        # 平均每个 turn: 买家消息 ~200 chars, 客服动作标记 ~50 chars
        dialogue_chars += 200 + 50  # buyer turn + agent turn

        history_lengths.append(dialogue_chars)

    return history_lengths


def build_prompt_for_step(
    playbook: Dict[str, Any],
    step: int,
    dialogue_chars: int,
    history_length: int = 5
) -> str:
    """
    构建第 step 步的 prompt
    """
    nodes = playbook.get('nodes', {})
    scenario = playbook.get('scenario', 'unknown')

    # 模拟当前节点（简化：假设始终有5-10个可用技能）
    available_skills = VALID_SKILLS[:10]  # 假设最多10个可用技能

    # 格式化字段
    available_skills_formatted = format_skills_with_desc(available_skills)
    skill_descriptions = format_skills_with_desc(available_skills[:10])
    slots_formatted = "- order_id: 12345\n- product_name: 测试商品"  # 模拟槽位

    # 模拟 buyer_text（取一个节点的文本作为示例）
    buyer_text = ""
    for node_id, node in nodes.items():
        if node.get('buyer_text'):
            buyer_text = node.get('buyer_text', '')
            break
    if not buyer_text:
        buyer_text = "这是买家的问题..."

    if step == 0:
        # 初始步，无历史
        prompt = CUSTOMER_SERVICE_TEMPLATE_NO_HIS.format(
            scenario=scenario,
            node_id='root',
            sentiment='neutral',
            buyer_text=buyer_text[:500],  # 限制长度
            slots_formatted=slots_formatted,
            available_skills_formatted=available_skills_formatted,
            skill_descriptions=skill_descriptions,
        )
    else:
        # 有历史
        # 格式化历史对话
        action_history = f"买家: {buyer_text[:200]}\n客服: [Action: gen_greet]\n" * min(step, history_length)

        prompt = CUSTOMER_SERVICE_TEMPLATE.format(
            task_description=f"处理{scenario}场景下的客户问题",
            step_count=step,
            history_length=min(history_length, step),
            action_history=action_history,
            scenario=scenario,
            node_id=f'node_{step}',
            current_step=step + 1,
            sentiment='neutral',
            buyer_text=buyer_text[:500],
            slots_formatted=slots_formatted,
            available_skills_formatted=available_skills_formatted,
            skill_descriptions=skill_descriptions,
        )

    return prompt


def analyze_actual_prompts(
    playbook_path: str,
    max_steps: int = 20,
    history_length: int = 5,
    max_prompt_length: int = 16384
):
    """
    分析实际 prompt 长度
    """
    print("=" * 70)
    print(" 真实 Prompt 长度分析报告")
    print("=" * 70)
    print(f"配置: max_steps={max_steps}, history_length={history_length}, max_prompt_length={max_prompt_length}")

    playbooks = load_playbooks(playbook_path)
    print(f"\n加载 {len(playbooks)} 个 playbooks")

    # 分析每个 playbook 在不同步骤的 prompt 长度
    max_prompt_chars = 0
    prompt_char_lengths = []
    prompt_token_lengths = []

    over_limit_count = 0
    over_limit_examples = []

    for pb in playbooks:
        nodes = pb.get('nodes', {})

        # 模拟对话历史增长
        history_lengths = simulate_dialogue_history(pb, max_steps)

        # 计算每个步骤的 prompt
        for step, dialogue_chars in enumerate(history_lengths):
            prompt = build_prompt_for_step(pb, step, dialogue_chars, history_length)
            char_len = len(prompt)
            token_len = estimate_tokens(char_len)

            prompt_char_lengths.append(char_len)
            prompt_token_lengths.append(token_len)

            if char_len > max_prompt_chars:
                max_prompt_chars = char_len

            if token_len > max_prompt_length:
                over_limit_count += 1
                if len(over_limit_examples) < 20:
                    over_limit_examples.append({
                        'session_id': pb.get('session_id', 'unknown'),
                        'scenario': pb.get('scenario', 'unknown'),
                        'nodes_count': len(nodes),
                        'step': step,
                        'char_len': char_len,
                        'token_len': token_len,
                        'exceeds_by': token_len - max_prompt_length,
                    })

    # 统计
    print("\n" + "=" * 70)
    print(" Prompt 长度统计")
    print("=" * 70)

    sorted_chars = sorted(prompt_char_lengths)
    sorted_tokens = sorted(prompt_token_lengths)
    n = len(sorted_chars)

    print(f"  总样本数: {n}")
    print(f"  最小字符: {sorted_chars[0]:,}")
    print(f"  最大字符: {sorted_chars[-1]:,}")
    print(f"  平均字符: {sum(sorted_chars) / n:,.1f}")
    print(f"  P90字符:  {sorted_chars[int(n * 0.90)]:,}")
    print(f"  P95字符:  {sorted_chars[int(n * 0.95)]:,}")
    print(f"  P99字符:  {sorted_chars[int(n * 0.99)]:,}")

    print(f"\n  最小Token: {sorted_tokens[0]:,}")
    print(f"  最大Token: {sorted_tokens[-1]:,}")
    print(f"  平均Token: {sum(sorted_tokens) / n:,.1f}")
    print(f"  P90Token:  {sorted_tokens[int(n * 0.90)]:,}")
    print(f"  P95Token:  {sorted_tokens[int(n * 0.95)]:,}")
    print(f"  P99Token:  {sorted_tokens[int(n * 0.99)]:,}")

    print("\n" + "=" * 70)
    print(f" 超出限制的样本 (max_prompt_length={max_prompt_length})")
    print("=" * 70)
    print(f"超出数量: {over_limit_count}")

    if over_limit_examples:
        print(f"\n超出限制示例 (前20个):")
        print(f"{'Session ID':<14} {'Scenario':<12} {'Nodes':<6} {'Step':<6} {'Chars':<10} {'Tokens':<10} {'超出':<8}")
        print("-" * 70)
        for ex in over_limit_examples:
            print(f"{ex['session_id'][:12]:<14} {ex['scenario']:<12} {ex['nodes_count']:<6} {ex['step']:<6} {ex['char_len']:<10,} {ex['token_len']:<10} +{ex['exceeds_by']:<8}")

    # 分析模板开销
    print("\n" + "=" * 70)
    print(" 模板固定开销分析")
    print("=" * 70)

    # 无历史模板的基础开销
    base_prompt = CUSTOMER_SERVICE_TEMPLATE_NO_HIS.format(
        scenario='presale',
        node_id='root',
        sentiment='neutral',
        buyer_text='测试消息',
        slots_formatted='（暂无）',
        available_skills_formatted=format_skills_with_desc(VALID_SKILLS[:10]),
        skill_descriptions=format_skills_with_desc(VALID_SKILLS[:10]),
    )
    print(f"基础模板开销 (无历史): {len(base_prompt):,} chars (~{estimate_tokens(len(base_prompt)):,} tokens)")

    # 有历史模板的基础开销（假设 5 步历史）
    history_prompt = CUSTOMER_SERVICE_TEMPLATE.format(
        task_description='测试任务',
        step_count=5,
        history_length=5,
        action_history='买家: 测试\n客服: [Action: test]\n' * 5,
        scenario='presale',
        node_id='node_5',
        current_step=6,
        sentiment='neutral',
        buyer_text='测试消息',
        slots_formatted='（暂无）',
        available_skills_formatted=format_skills_with_desc(VALID_SKILLS[:10]),
        skill_descriptions=format_skills_with_desc(VALID_SKILLS[:10]),
    )
    print(f"带历史模板开销 (5步): {len(history_prompt):,} chars (~{estimate_tokens(len(history_prompt)):,} tokens)")

    # 分析技能描述开销
    skills_10 = format_skills_with_desc(VALID_SKILLS[:10])
    skills_31 = format_skills_with_desc(VALID_SKILLS)
    print(f"技能描述开销 (10技能): {len(skills_10):,} chars (~{estimate_tokens(len(skills_10)):,} tokens)")
    print(f"技能描述开销 (31技能): {len(skills_31):,} chars (~{estimate_tokens(len(skills_31)):,} tokens)")

    # 分析对话历史增长
    print("\n" + "=" * 70)
    print(" 对话历史增长分析")
    print("=" * 70)
    print("假设每步增长: 买家~200 chars + 客服~50 chars = 250 chars/step")
    print(f"history_length={history_length} 限制下，最大历史长度: ~{history_length * 250:,} chars")
    print(f"max_steps={max_steps} 步后的累积历史（无截断）: ~{max_steps * 250:,} chars")

    # 结论
    print("\n" + "=" * 70)
    print(" 结论")
    print("=" * 70)

    if max_prompt_chars > 0:
        max_tokens = estimate_tokens(max_prompt_chars)
        print(f"1. 最大 Prompt 字符数: {max_prompt_chars:,}")
        print(f"2. 最大 Prompt Token估算: {max_tokens:,}")
        print(f"3. 配置 max_prompt_length: {max_prompt_length}")

        if max_tokens > max_prompt_length:
            print(f"❌ 超出限制！需要: {max_tokens:,} > 配置: {max_prompt_length}")
            print(f"   建议: 增加 max_prompt_length 或 减少 history_length/max_steps")
        else:
            print(f"✅ 在限制范围内: {max_tokens:,} <= {max_prompt_length}")

    print("\n" + "=" * 70)


def analyze_parquet_prompts(parquet_path: str):
    """
    分析 parquet 文件中存储的初始 prompt
    """
    print("\n" + "=" * 70)
    print(" Parquet 文件初始 Prompt 分析")
    print("=" * 70)

    df = pd.read_parquet(parquet_path)

    prompt_lengths = []
    for idx, row in df.iterrows():
        prompt = row['prompt']
        # prompt 是一个 list of dicts: [{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}]
        total_chars = 0
        for p in prompt:
            content = p.get('content', '')
            total_chars += len(content)
        prompt_lengths.append(total_chars)

    sorted_lengths = sorted(prompt_lengths)
    n = len(sorted_lengths)

    print(f"  总样本数: {n}")
    print(f"  最小字符: {sorted_lengths[0]:,}")
    print(f"  最大字符: {sorted_lengths[-1]:,}")
    print(f"  平均字符: {sum(sorted_lengths) / n:,.1f}")
    print(f"  P90字符:  {sorted_lengths[int(n * 0.90)]:,}")
    print(f"  P95字符:  {sorted_lengths[int(n * 0.95)]:,}")
    print(f"  P99字符:  {sorted_lengths[int(n * 0.99)]:,}")
    print(f"  估算Token: {estimate_tokens(sorted_lengths[-1]):,} (最大)")

    # 打印示例
    sample_idx = sorted_lengths[-1]  # 最大长度的索引
    max_idx = prompt_lengths.index(sorted_lengths[-1])
    sample = df.iloc[max_idx]['prompt']
    print(f"\n  最大 Prompt 示例 (前1000字符):")
    total_content = ""
    for p in sample:
        total_content += p.get('content', '')
    print(f"  {total_content[:1000]}...")


def main():
    playbook_path = 'outputs/playbooks_full.json'
    parquet_train_path = os.path.expanduser("~/data/verl-agent/customer_service/train.parquet")
    parquet_test_path = os.path.expanduser("~/data/verl-agent/customer_service/test.parquet")

    # 分析 playbooks
    analyze_actual_prompts(
        playbook_path,
        max_steps=20,
        history_length=5,
        max_prompt_length=16384
    )

    # 分析 parquet 文件
    if os.path.exists(parquet_train_path):
        analyze_parquet_prompts(parquet_train_path)

    if os.path.exists(parquet_test_path):
        analyze_parquet_prompts(parquet_test_path)


if __name__ == '__main__':
    main()