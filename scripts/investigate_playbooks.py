#!/usr/bin/env python3
"""
Playbook 数据探索性分析脚本

用于分析 playbooks_full.json 中的数据，找出可能导致 Prompt 超长的异常数据。

Usage:
    python scripts/investigate_playbooks.py
"""

import json
import os
import statistics
from typing import Dict, List, Any, Tuple
from collections import defaultdict


def load_playbooks(path: str) -> List[Dict[str, Any]]:
    """加载 playbook 数据"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_static_context_length(playbook: Dict[str, Any]) -> int:
    """
    计算静态上下文的字符长度

    静态上下文包括：
    - scenario, subtype
    - initial_slots, business_outcome
    - 所有 nodes 的完整内容（序列化后的JSON字符串长度）
    """
    total_chars = 0

    # 1. 元数据
    scenario = playbook.get('scenario', '')
    subtype = playbook.get('subtype', '')
    total_chars += len(scenario) + len(subtype)

    # 2. initial_slots 和 business_outcome
    initial_slots = playbook.get('initial_slots', {})
    business_outcome = playbook.get('business_outcome', {})
    total_chars += len(json.dumps(initial_slots, ensure_ascii=False))
    total_chars += len(json.dumps(business_outcome, ensure_ascii=False))

    # 3. 所有 nodes 的内容
    nodes = playbook.get('nodes', {})
    for node_id, node in nodes.items():
        # node_id
        total_chars += len(node_id)

        # buyer_text (通常是主要长度来源)
        buyer_text = node.get('buyer_text', '')
        total_chars += len(buyer_text)

        # available_skills
        available_skills = node.get('available_skills', [])
        total_chars += sum(len(s) for s in available_skills)

        # transitions (skill -> node_id 映射)
        transitions = node.get('transitions', {})
        for skill, target in transitions.items():
            total_chars += len(skill) + len(target)

        # sentiment, slot_updates, default_fallback
        sentiment = node.get('sentiment', '')
        slot_updates = node.get('slot_updates', {})
        default_fallback = node.get('default_fallback', '')
        total_chars += len(sentiment)
        total_chars += len(json.dumps(slot_updates, ensure_ascii=False))
        total_chars += len(default_fallback)

    return total_chars


def count_nodes(playbook: Dict[str, Any]) -> int:
    """统计节点数量"""
    return len(playbook.get('nodes', {}))


def estimate_tokens(char_count: int) -> int:
    """
    粗略估算 token 数量

    中文: 约 1.5 字符/token
    英文/数字/符号: 约 4 字符/token
    平均估算: 约 2 字符/token
    """
    return char_count // 2


def calculate_statistics(values: List[int]) -> Dict[str, float]:
    """计算统计指标"""
    if not values:
        return {}

    sorted_values = sorted(values)
    n = len(sorted_values)

    return {
        'count': n,
        'mean': statistics.mean(sorted_values),
        'median': statistics.median(sorted_values),
        'min': sorted_values[0],
        'max': sorted_values[-1],
        'p90': sorted_values[int(n * 0.90)] if n > 0 else 0,
        'p95': sorted_values[int(n * 0.95)] if n > 0 else 0,
        'p99': sorted_values[int(n * 0.99)] if n > 0 else 0,
        'std': statistics.stdev(sorted_values) if n > 1 else 0,
    }


def print_stats_table(title: str, stats: Dict[str, float], unit: str = "") -> None:
    """打印统计表格"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"  总数量:     {stats['count']:,}")
    print(f"  最小值:     {stats['min']:,}{unit}")
    print(f"  最大值:     {stats['max']:,}{unit}")
    print(f"  平均值:     {stats['mean']:,.1f}{unit}")
    print(f"  中位数:     {stats['median']:,.1f}{unit}")
    print(f"  标准差:     {stats['std']:,.1f}{unit}")
    print(f"  P90:        {stats['p90']:,.0f}{unit}")
    print(f"  P95:        {stats['p95']:,.0f}{unit}")
    print(f"  P99:        {stats['p99']:,.0f}{unit}")


def find_top_outliers(
    playbooks: List[Dict[str, Any]],
    metric_fn,
    top_n: int = 5
) -> List[Tuple[Dict[str, Any], int]]:
    """找出指标最大的 top_n 个 playbook"""
    results = []
    for pb in playbooks:
        value = metric_fn(pb)
        results.append((pb, value))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def print_outliers(title: str, outliers: List[Tuple[Dict[str, Any], int]], metric_name: str) -> None:
    """打印异常值"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"{'排名':<4} {'Session ID':<14} {'Scenario':<12} {'Nodes':<6} {metric_name:<12} {'Has Order':<10}")
    print("-" * 70)

    for i, (pb, value) in enumerate(outliers, 1):
        session_id = pb.get('session_id', 'N/A')[:12]
        scenario = pb.get('scenario', 'unknown')
        nodes_count = len(pb.get('nodes', {}))
        has_order = pb.get('business_outcome', {}).get('has_order', False)
        order_amount = pb.get('business_outcome', {}).get('order_amount', 0)

        order_info = f"Y(¥{order_amount:.0f})" if has_order else "N"
        print(f"{i:<4} {session_id:<14} {scenario:<12} {nodes_count:<6} {value:<12,} {order_info:<10}")


def analyze_by_scenario(playbooks: List[Dict[str, Any]]) -> None:
    """按场景分析"""
    print(f"\n{'='*60}")
    print(" 按场景分析")
    print(f"{'='*60}")

    scenario_stats = defaultdict(lambda: {'count': 0, 'total_chars': 0, 'total_nodes': 0, 'char_values': [], 'node_values': []})

    for pb in playbooks:
        scenario = pb.get('scenario', 'unknown')
        char_len = calculate_static_context_length(pb)
        node_count = count_nodes(pb)

        scenario_stats[scenario]['count'] += 1
        scenario_stats[scenario]['total_chars'] += char_len
        scenario_stats[scenario]['total_nodes'] += node_count
        scenario_stats[scenario]['char_values'].append(char_len)
        scenario_stats[scenario]['node_values'].append(node_count)

    print(f"\n{'场景':<12} {'数量':<8} {'平均字符':<12} {'最大字符':<12} {'平均节点':<10} {'最大节点':<10}")
    print("-" * 70)

    for scenario in sorted(scenario_stats.keys(), key=lambda x: scenario_stats[x]['count'], reverse=True):
        stats = scenario_stats[scenario]
        avg_chars = stats['total_chars'] / stats['count'] if stats['count'] > 0 else 0
        max_chars = max(stats['char_values']) if stats['char_values'] else 0
        avg_nodes = stats['total_nodes'] / stats['count'] if stats['count'] > 0 else 0
        max_nodes = max(stats['node_values']) if stats['node_values'] else 0

        print(f"{scenario:<12} {stats['count']:<8} {avg_chars:<12,.0f} {max_chars:<12,} {avg_nodes:<10,.1f} {max_nodes:<10}")


def analyze_long_text_nodes(playbooks: List[Dict[str, Any]], threshold: int = 500) -> None:
    """分析包含超长文本节点的 playbook"""
    print(f"\n{'='*60}")
    print(f" 超长文本节点分析 (buyer_text > {threshold} 字符)")
    print(f"{'='*60}")

    long_text_playbooks = []

    for pb in playbooks:
        nodes = pb.get('nodes', {})
        max_text_len = 0
        long_node_id = ''

        for node_id, node in nodes.items():
            buyer_text = node.get('buyer_text', '')
            if len(buyer_text) > max_text_len:
                max_text_len = len(buyer_text)
                long_node_id = node_id

        if max_text_len > threshold:
            long_text_playbooks.append({
                'playbook': pb,
                'max_text_len': max_text_len,
                'node_id': long_node_id
            })

    long_text_playbooks.sort(key=lambda x: x['max_text_len'], reverse=True)

    print(f"包含超长文本节点的 playbook 数量: {len(long_text_playbooks)}")

    if long_text_playbooks:
        print(f"\nTop 10 超长文本节点:")
        print(f"{'排名':<4} {'Session ID':<14} {'Scenario':<12} {'节点ID':<20} {'文本长度':<10}")
        print("-" * 70)

        for i, item in enumerate(long_text_playbooks[:10], 1):
            pb = item['playbook']
            print(f"{i:<4} {pb.get('session_id', 'N/A')[:12]:<14} {pb.get('scenario', 'unknown'):<12} {item['node_id']:<20} {item['max_text_len']:<10,}")


def check_token_limit(playbooks: List[Dict[str, Any]], max_prompt_length: int = 8192) -> None:
    """检查可能超出 token 限制的 playbook"""
    print(f"\n{'='*60}")
    print(f" Token 限制检查 (max_prompt_length = {max_prompt_length})")
    print(f"{'='*60}")

    # 估算：静态上下文 + system prompt + 其他开销
    # 假设 system prompt 和其他开销约 500 tokens
    SYSTEM_PROMPT_OVERHEAD = 500

    over_limit = []
    near_limit = []

    for pb in playbooks:
        char_len = calculate_static_context_length(pb)
        estimated_tokens = estimate_tokens(char_len) + SYSTEM_PROMPT_OVERHEAD

        if estimated_tokens > max_prompt_length:
            over_limit.append({
                'playbook': pb,
                'char_len': char_len,
                'estimated_tokens': estimated_tokens,
                'exceeds_by': estimated_tokens - max_prompt_length
            })
        elif estimated_tokens > max_prompt_length * 0.9:
            near_limit.append({
                'playbook': pb,
                'char_len': char_len,
                'estimated_tokens': estimated_tokens
            })

    print(f"超出限制的 playbook: {len(over_limit)}")
    print(f"接近限制 (90%+) 的 playbook: {len(near_limit)}")

    if over_limit:
        over_limit.sort(key=lambda x: x['exceeds_by'], reverse=True)
        print(f"\n超出限制的 playbook 详情:")
        print(f"{'Session ID':<14} {'Scenario':<12} {'节点数':<8} {'字符数':<10} {'估算Tokens':<12} {'超出':<8}")
        print("-" * 70)

        for item in over_limit[:20]:
            pb = item['playbook']
            print(f"{pb.get('session_id', 'N/A')[:12]:<14} {pb.get('scenario', 'unknown'):<12} {len(pb.get('nodes', {})):<8} {item['char_len']:<10,} {item['estimated_tokens']:<12} +{item['exceeds_by']:<8}")


def main():
    """主函数"""
    print("=" * 60)
    print(" Playbook 数据探索性分析报告")
    print("=" * 60)

    # 加载数据
    playbook_path = 'outputs/playbooks_full.json'
    print(f"\n加载数据: {playbook_path}")
    playbooks = load_playbooks(playbook_path)
    print(f"总 playbook 数量: {len(playbooks)}")

    # 1. 静态上下文长度分析
    char_lengths = []
    for pb in playbooks:
        char_len = calculate_static_context_length(pb)
        char_lengths.append(char_len)

    char_stats = calculate_statistics(char_lengths)
    print_stats_table("静态上下文字符长度分布", char_stats, " 字符")

    # Token 估算
    token_estimates = [estimate_tokens(c) for c in char_lengths]
    token_stats = calculate_statistics(token_estimates)
    print_stats_table("静态上下文估算 Token 分布", token_stats, " tokens")

    # 2. 节点数分布
    node_counts = [count_nodes(pb) for pb in playbooks]
    node_stats = calculate_statistics(node_counts)
    print_stats_table("节点数分布", node_stats, " 个")

    # 3. 找出异常值 - 字符长度 Top 5
    length_outliers = find_top_outliers(playbooks, calculate_static_context_length, top_n=5)
    print_outliers("字符长度 Top 5 异常值", length_outliers, "字符数")

    # 4. 找出异常值 - 节点数 Top 5
    node_outliers = find_top_outliers(playbooks, count_nodes, top_n=5)
    print_outliers("节点数 Top 5 异常值", node_outliers, "节点数")

    # 5. 按场景分析
    analyze_by_scenario(playbooks)

    # 6. 超长文本节点分析
    analyze_long_text_nodes(playbooks, threshold=500)

    # 7. Token 限制检查
    check_token_limit(playbooks, max_prompt_length=8192)

    # 总结
    print(f"\n{'='*60}")
    print(" 分析总结")
    print(f"{'='*60}")
    print(f"1. 总 playbook 数量: {len(playbooks)}")
    print(f"2. 静态上下文字符长度: 平均 {char_stats['mean']:,.0f}, 最大 {char_stats['max']:,}")
    print(f"3. 估算 Token 数量: 平均 {token_stats['mean']:,.0f}, 最大 {token_stats['max']:,}")
    print(f"4. 节点数: 平均 {node_stats['mean']:.1f}, 最大 {int(node_stats['max'])}")
    print(f"5. P99 字符长度: {char_stats['p99']:,.0f} (约 {estimate_tokens(char_stats['p99']):,} tokens)")
    print(f"6. P99 节点数: {int(node_stats['p99'])}")

    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()