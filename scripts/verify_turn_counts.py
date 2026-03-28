#!/usr/bin/env python3
"""
验证 effective_turn_count 回填正确性的脚本

对抽样样本进行逐条验证：
1. 加载原始 session 消息
2. 手动执行清洗逻辑（aggregate_turns + Agent-first 处理）
3. 对比计算值与回填值
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.aggregator import aggregate_turns, ROLE_MAPPING
from etl.cleaner import validate_user_agent_alternation


def load_raw_sessions(file_path: str) -> dict:
    """加载原始 session 数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict) and 'data' in raw_data:
        return {s['session_id']: s for s in raw_data['data']}
    elif isinstance(raw_data, list):
        return {s['session_id']: s for s in raw_data}
    return {}


def calculate_effective_turn_count_verbose(messages: list, session_id: str = "unknown") -> dict:
    """
    计算有效对话轮次（详细版本，输出中间步骤）

    Returns:
        dict with turn_count, steps, and intermediate turns
    """
    result = {
        'session_id': session_id,
        'raw_message_count': len(messages),
        'steps': [],
        'turns_after_aggregation': 0,
        'turns_after_agent_removal': 0,
        'final_turn_count': 0,
        'turns_detail': [],
        'match': False,
    }

    # Step 1: 统计原始消息的角色分布
    from collections import Counter
    role_dist = Counter(m.get('sent_by', 'unknown') for m in messages)
    result['raw_role_distribution'] = dict(role_dist)
    result['steps'].append(f"原始消息: {len(messages)} 条, 角色分布: {dict(role_dist)}")

    # Step 2: 执行 aggregate_turns
    try:
        aggregated = aggregate_turns(messages)
        turns = aggregated['turns']
        result['turns_after_aggregation'] = len(turns)
        result['steps'].append(f"聚合后: {len(turns)} turns")
    except Exception as e:
        result['steps'].append(f"聚合失败: {e}")
        return result

    # Step 3: 处理以 Agent 开头的 session
    if turns and turns[0]['role'] == 'Agent':
        first_user_idx = None
        for i, turn in enumerate(turns):
            if turn['role'] == 'User':
                first_user_idx = i
                break

        if first_user_idx is not None:
            removed_count = first_user_idx
            turns = turns[first_user_idx:]
            result['steps'].append(f"删除开头 {removed_count} 个 Agent turns")
        else:
            result['steps'].append("全是 Agent 消息，无效 session")
            turns = []

    result['turns_after_agent_removal'] = len(turns)

    # Step 4: 验证交替
    if turns:
        is_valid = validate_user_agent_alternation(turns)
        result['steps'].append(f"交替验证: {'通过' if is_valid else '失败'}")

    # Step 5: 记录最终 turns 详情
    result['final_turn_count'] = len(turns)
    result['turns_detail'] = [
        {'idx': i, 'role': t['role'], 'text_preview': t.get('text', '')[:50] + '...'}
        for i, t in enumerate(turns)
    ]

    return result


def verify_samples(sample_playbooks: list, raw_sessions: dict) -> list:
    """验证样本"""
    results = []

    for pb in sample_playbooks:
        session_id = pb.get('session_id')
        expected_count = pb.get('effective_turn_count')

        raw_session = raw_sessions.get(session_id)
        if raw_session is None:
            results.append({
                'session_id': session_id,
                'error': 'Raw session not found',
                'match': False,
            })
            continue

        messages = raw_session.get('messages', [])
        calc_result = calculate_effective_turn_count_verbose(messages, session_id)

        calc_result['expected_count'] = expected_count
        calc_result['match'] = (calc_result['final_turn_count'] == expected_count)

        results.append(calc_result)

    return results


def print_verification_report(results: list):
    """打印验证报告"""
    print("=" * 70)
    print(" 验证报告：effective_turn_count 回填正确性")
    print("=" * 70)

    matches = sum(1 for r in results if r.get('match', False))
    total = len(results)

    print(f"\n总体验证结果: {matches}/{total} 通过")

    if matches == total:
        print("✅ 所有样本验证通过！")
    else:
        print("❌ 部分样本验证失败，请检查")

    print("\n" + "-" * 70)
    print(" 详细验证过程")
    print("-" * 70)

    for r in results:
        session_id = r.get('session_id', 'unknown')
        match = "✅" if r.get('match') else "❌"
        expected = r.get('expected_count', '?')
        calculated = r.get('final_turn_count', '?')

        print(f"\n【{match}】{session_id}")
        print(f"  期望值: {expected}, 计算值: {calculated}")

        # 打印步骤
        for step in r.get('steps', []):
            print(f"  - {step}")

        # 打印角色分布
        if 'raw_role_distribution' in r:
            print(f"  原始角色分布: {r['raw_role_distribution']}")

        # 如果不匹配，打印 turns 详情
        if not r.get('match') and r.get('turns_detail'):
            print(f"  Turns 详情 (前10条):")
            for t in r['turns_detail'][:10]:
                print(f"    {t['idx']}. [{t['role']}]: {t['text_preview']}")

    print("\n" + "=" * 70)


def main():
    # 加载样本
    sample_path = '/tmp/sample_playbooks.json'
    if not Path(sample_path).exists():
        print(f"错误: 样本文件不存在 {sample_path}")
        print("请先运行抽样脚本生成样本")
        return

    with open(sample_path, 'r') as f:
        samples = json.load(f)

    print(f"加载 {len(samples)} 个样本")

    # 加载原始 session 数据
    raw_sessions = load_raw_sessions('session_order_converted.json')
    print(f"加载 {len(raw_sessions)} 个原始 session")

    # 验证
    results = verify_samples(samples, raw_sessions)

    # 打印报告
    print_verification_report(results)

    # 保存详细结果
    with open('/tmp/verification_results.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存到 /tmp/verification_results.json")


if __name__ == '__main__':
    main()