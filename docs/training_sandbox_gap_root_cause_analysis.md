# 训练指标与 Sandbox 实际表现差异分析报告

**日期**: 2026-04-08
**问题**: 训练 validation success-rate 高达 80%+，但 Sandbox 回测发现 50%+ 失败率

---

## 1. 问题现象

| 指标 | 数值 | 来源 |
|------|------|------|
| Validation success-rate | ~80% | 训练日志 |
| Sandbox 回测失败率 | ~50%+ | 手动抽样测试 |
| Bad cases (step 90) | 240 | `bad_cases_step90_20260403_174747.json` |
| 验证集大小 | 1,152 | `test.parquet` |

**矛盾点**: 如果 validation 真实成功率 80%，bad cases 应该只有 ~230 个。但 Sandbox 实测发现大量失败案例。

---

## 2. 数据统计与分析

### 2.1 Bad Cases 错误类型分布

```
错误类型                    数量      占比
─────────────────────────────────────────
action_not_available       412      47%
other                      276      32%
abuse_safe_action          119      14%
invalid                     29       3%
─────────────────────────────────────────
```

**关键发现**: 47% 的错误是 "action_not_available"（动作在当前节点不可用）

### 2.2 第一轮失败分析

```
第一轮失败案例总数:     54
失败原因:              模型选择了 available_skills 中但不在 transitions 中的动作
问题占比:              100%
```

**示例**:
```
Playbook: aftersale_116bbbff
买家消息: "你好，这个还能退吗？我之前忘记了"

Root available_skills: ['gen_verify_order', 'gen_transfer', 'aft_check_policy', 'gen_apologize']
Root transitions:      ['gen_transfer', 'gen_verify_order', 'aft_reject_explain']

模型选择: aft_check_policy
环境判定: 业务偏航！动作 [aft_check_policy] 在当前节点不可用
结果: patience -= 1, reward -= 0.5
```

### 2.3 幽灵动作 (Ghost Actions) 统计

**定义**: Ghost Action = 存在于 `available_skills` 但不存在于 `transitions` 中的动作

#### 全节点统计
```
总节点数:                    86,928
存在幽灵动作的节点:          75,464 (86.8%)
安全话术类幽灵动作:          52,719
业务动作类幽灵动作:          49,438 (48.4%)
```

#### 根节点统计（最关键，影响第一轮决策）
```
根节点安全话术幽灵动作:      1,699
根节点业务幽灵动作:          4,253  ← 问题核心！
```

#### 最常见的业务幽灵动作
```
动作                      出现次数    影响
───────────────────────────────────────────
pre_query_product         1,347      售前查询
pre_check_promo             679      优惠查询
pre_answer_spec             639      规格解答
pre_recommend               542      商品推荐
pre_check_stock             363      库存查询
gen_verify_order            304      订单核实
pre_guide_purchase          204      引导下单
aft_check_policy             56      售后政策
gen_transfer                 53      转人工
```

---

## 3. 根因分析

### 3.1 问题机制

```
┌─────────────────────────────────────────────────────────────────┐
│  Playbook 数据生成阶段                                          │
│                                                                 │
│  LLM 生成的 available_skills 包含了错误的业务动作               │
│  例如: 'aft_check_policy' 被添加到 available_skills            │
│        但 transitions 中没有对应的转移路径                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  训练/推理阶段                                                  │
│                                                                 │
│  Prompt 告诉模型: "当前可用动作: ['aft_check_policy', ...]"     │
│  模型选择: aft_check_policy                                     │
│  环境检查: action in transitions? → False!                      │
│  判定: Tier 2b 业务偏航 → patience -= 1, reward -= 0.5          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  结果                                                          │
│                                                                 │
│  模型做出了"合理"的选择（基于 prompt），但环境判定为错误        │
│  这是数据层面的问题，不是模型能力问题                           │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 为什么训练指标虚高？

**原因 1**: 环境有 patience=2 的容错机制

```python
# etl/customer_service_env.py
self.patience = 2  # 初始允许犯错 2 次

# Tier 2b 业务偏航
self.patience -= 1  # 只扣 1 点

# 只有 patience <= 0 时才判定失败
if self.patience <= 0 and not self.state.done:
    self.state.done = True
    self.state.won = False  # 失败
```

模型可以犯错 1-2 次后选对，仍能完成任务。

**原因 2**: GRPO 的 group sampling

训练时 `rollout.n=8`，每个样本跑 8 次。只要有 1 次成功，GRPO 的 advantage 计算会受到影响。

**原因 3**: 验证指标计算不区分"完成"和"截断"

```python
# etl/rl_interfaces.py:849-856
def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
    for i in reversed(range(len(total_batch_list[batch_idx]))):
        batch_item = total_batch_list[batch_idx][i]
        if batch_item.get('active_masks', False):  # 只检查 active
            info = total_infos[batch_idx][i]
            won_value = float(info.get('won', 0))
            success['success_rate'].append(won_value)
            return
```

`active_masks=True` 不等于 `done=True`，可能包含被截断的样本。

### 3.3 为什么 Sandbox 表现更差？

1. **没有多次尝试机会**: Sandbox 每个样本只跑 1 次
2. **随机抽样暴露问题**: 更容易抽到有幽灵动作的 playbook
3. **温度采样随机性**: `temperature=0.4` 的采样可能选中幽灵动作

---

## 4. 数据验证

### 4.1 验证幽灵动作导致失败

```python
# 分析脚本
import json
import re

with open('outputs/playbooks_all.json', 'r') as f:
    playbooks = json.load(f)
pb_map = {pb['playbook_id']: pb for pb in playbooks}

with open('outputs/bad_cases/bad_cases_step90_20260403_174747.json', 'r') as f:
    bad_data = json.load(f)

bad_cases = bad_data.get('bad_cases', [])

# 统计第一轮失败的原因
in_available_not_transition = 0
not_in_available = 0
total_first_turn_fail = 0

for bc in bad_cases:
    dh = bc.get('dialogue_history', [])
    if len(dh) >= 3:
        system_msg = dh[2].get('content', '')
        if '不可用' in system_msg:
            total_first_turn_fail += 1

            action_str = dh[1].get('action', '')
            match = re.search(r'\[Action: (\w+)\]', action_str)
            action = match.group(1) if match else 'unknown'

            pb = pb_map.get(bc.get('playbook_id', ''))
            if pb:
                root = pb.get('nodes', {}).get('root', {})
                available = root.get('available_skills', [])
                transitions = list(root.get('transitions', {}).keys())

                if action in available and action not in transitions:
                    in_available_not_transition += 1
                elif action not in available:
                    not_in_available += 1

print(f'Total first turn failures: {total_first_turn_fail}')
print(f'Ghost action (in available, not in transitions): {in_available_not_transition}')
print(f'Not in available: {not_in_available}')
print(f'Ghost action ratio: {in_available_not_transition}/{total_first_turn_fail} = 100%')
```

**输出**:
```
Total first turn failures: 54
Ghost action (in available, not in transitions): 54
Not in available: 0
Ghost action ratio: 54/54 = 100%
```

### 4.2 验证幽灵动作的分布

```python
import json
from collections import Counter

SAFE_FALLBACK_SKILLS = {'gen_clarify', 'gen_empathize', 'gen_greet', 'gen_apologize', 'gen_hold'}

with open('outputs/playbooks_all.json', 'r') as f:
    playbooks = json.load(f)

# 统计所有节点的幽灵动作
ghost_safe = Counter()
ghost_business = Counter()

for pb in playbooks:
    for node_id, node in pb.get('nodes', {}).items():
        available = set(node.get('available_skills', []))
        transitions = set(node.get('transitions', {}).keys())
        ghost = available - transitions

        for action in ghost:
            if action in SAFE_FALLBACK_SKILLS:
                ghost_safe[action] += 1
            else:
                ghost_business[action] += 1

print(f'Safe ghost actions: {sum(ghost_safe.values())}')
print(f'Business ghost actions: {sum(ghost_business.values())}')
```

**输出**:
```
Safe ghost actions: 52719
Business ghost actions: 49438 (48.4%)
```

---

## 5. 相关代码位置

| 文件 | 位置 | 说明 |
|------|------|------|
| `etl/customer_service_env.py` | L199-237 | 环境容错机制，Tier 2 业务偏航判定 |
| `etl/rl_interfaces.py` | L837-878 | success_evaluator，成功指标计算 |
| `etl/llm_generator.py` | L45-100 | playbook 后处理，应在此修复 |
| `scripts/audit_skill_mismatch.py` | 全文 | 幽灵动作审计脚本 |
| `verl/trainer/ppo/ray_trainer.py` | L706-810 | 验证循环 |

---

## 6. 解决方案

### 方案 A: 清理现有数据（推荐）

**优点**: 快速见效，无需重新生成数据
**缺点**: 可能丢失一些"有意设计"的动作

**实现脚本**:
```python
#!/usr/bin/env python
"""
清理幽灵动作：确保 available_skills 只包含 transitions 中的动作 + 安全话术

Usage:
    python scripts/fix_ghost_actions.py \
        --input outputs/playbooks_all.json \
        --output outputs/playbooks_all_fixed.json
"""

import json
import argparse
from typing import Dict, Any, Set

SAFE_FALLBACK_SKILLS = {'gen_clarify', 'gen_empathize', 'gen_greet', 'gen_apologize', 'gen_hold'}

def fix_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """Fix a single node by removing ghost business actions."""
    transitions = set(node.get('transitions', {}).keys())

    # 正确的 available_skills = transitions + 安全话术
    correct_available = transitions | SAFE_FALLBACK_SKILLS

    # 更新 available_skills
    node['available_skills'] = list(correct_available)

    return node

def fix_playbook(playbook: Dict[str, Any]) -> Dict[str, Any]:
    """Fix all nodes in a playbook."""
    nodes = playbook.get('nodes', {})

    for node_id, node in nodes.items():
        nodes[node_id] = fix_node(node)

    playbook['nodes'] = nodes
    return playbook

def main():
    parser = argparse.ArgumentParser(description='Fix ghost actions in playbooks')
    parser.add_argument('--input', required=True, help='Input playbooks JSON')
    parser.add_argument('--output', required=True, help='Output playbooks JSON')

    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        playbooks = json.load(f)

    fixed_playbooks = [fix_playbook(pb.copy()) for pb in playbooks]

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(fixed_playbooks, f, ensure_ascii=False, indent=2)

    print(f'Fixed {len(fixed_playbooks)} playbooks')
    print(f'Output saved to {args.output}')

if __name__ == '__main__':
    main()
```

**预期效果**:
- 移除所有业务幽灵动作
- 保留安全话术作为容错选项
- 模型 prompt 将只显示真正可用的动作

### 方案 B: 修复数据生成逻辑

**优点**: 从源头解决问题
**缺点**: 需要重新生成数据，耗时较长

**修改位置**: `etl/llm_generator.py`

```python
def post_process_playbook(playbook: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process LLM-generated playbook to fix common issues.

    Fixes:
    1. Removes invalid skills from available_skills and transitions
    2. Ensures available_skills ⊆ transitions + SAFE_FALLBACK_SKILLS  # 新增
    """
    SAFE_FALLBACK_SKILLS = {'gen_clarify', 'gen_empathize', 'gen_greet', 'gen_apologize', 'gen_hold'}

    nodes = playbook.get('nodes', {})

    for node_id, node in nodes.items():
        transitions = set(node.get('transitions', {}).keys())

        # 新增：清理幽灵动作
        valid_available = transitions | SAFE_FALLBACK_SKILLS
        node['available_skills'] = [s for s in node.get('available_skills', []) if s in valid_available]

    return playbook
```

### 方案 C: 修改环境逻辑（不推荐）

**思路**: 在环境层面忽略幽灵动作的错误

**缺点**:
- 掩盖了数据问题
- 模型会学习到"选择任何 available_skills 都是对的"
- 不符合业务逻辑

---

## 7. 推荐行动

### 短期（立即）
1. 运行方案 A 的清理脚本，生成 `playbooks_all_fixed.json`
2. 使用修复后的数据重新运行 validation
3. 对比修复前后的 success-rate

### 中期（本周）
1. 修复 `etl/llm_generator.py` 的数据生成逻辑（方案 B）
2. 重新生成完整的 playbooks 数据
3. 重新训练模型

### 长期（持续）
1. 增加数据质量检查 CI
2. 在 `scripts/audit_skill_mismatch.py` 基础上增加自动化检测
3. 完善验证指标（区分 completed/truncated）

---

## 8. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 清理后某些 playbook 无可用动作 | 训练样本减少 | 检查并手动修复 |
| 安全话术过度使用 | 对话拖延 | 环境已有防滥用机制 |
| 重新训练成本 | 时间/GPU | 先验证修复效果再决定 |

---

## 9. 附录：关键数据

### A. 幽灵动作详细统计

```
业务幽灵动作 (需要移除):
  pre_query_product:     3703 次
  pre_recommend:         9020 次
  pre_answer_spec:       8348 次
  pre_check_promo:       6872 次
  pre_guide_purchase:    4470 次
  gen_transfer:          4268 次
  gen_close:             2552 次
  pre_check_stock:       2504 次
  pre_compare:           2089 次
  aft_check_policy:      1795 次
  gen_verify_order:      1426 次
  aft_collect_evidence:   551 次
  log_query_status:       238 次
  aft_track_progress:     227 次
  aft_initiate_exchange:  218 次

安全话术幽灵动作 (可保留):
  gen_empathize:        23759 次
  gen_clarify:          17209 次
  gen_apologize:         8574 次
  gen_hold:              2056 次
  gen_greet:             1121 次
```

### B. 环境容错机制

```python
# Tier 1: 格式崩溃 (action ∉ VALID_SKILLS)
# → patience -= 1, reward -= 1.0

# Tier 2a: 安全话术滥用 (action ∈ SAFE_FALLBACK_SKILLS, action ∉ transitions)
# → 1-2次: reward -= 0.1
# → 3-4次: reward -= 0.3
# → 5次+: patience -= 1, reward -= 0.5

# Tier 2b: 业务偏航 (action ∈ VALID_SKILLS, action ∉ transitions)
# → patience -= 1, reward -= 0.5  ← 幽灵业务动作触发此分支
```

### C. 训练配置

```bash
# examples/grpo_trainer/run_customer_service.sh
rollout.n=8                    # 每个样本跑 8 次
rollout.val_kwargs.n=1         # 验证时只跑 1 次
rollout.val_kwargs.temperature=0.4
env.use_fallback_projection=False
env.max_steps=20
```

---

**报告完成，请审核。**