# 训练指标与实际表现差异分析报告

## 问题描述

训练日志显示 validation success-rate 高达 80%+，但通过 sandbox 回测发现至少一半以上无法成功走完整个 session。

---

## 分析结论

### 根本原因：验证指标计算存在幸存者偏差

**核心问题**：验证时的 success_rate 只计算了**完整跑完**的 episode，忽略了中途被截断的失败样本。

```python
# etl/rl_interfaces.py:849-878
def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
    for i in reversed(range(len(total_batch_list[batch_idx]))):
        batch_item = total_batch_list[batch_idx][i]
        if batch_item.get('active_masks', False):  # <-- 只处理 active 的样本
            info = total_infos[batch_idx][i]
            won_value = float(info.get('won', 0))
            success['success_rate'].append(won_value)
            return
```

**问题**：`active_masks=True` 的样本不一定是成功完成的，可能是：
1. 正常完成（won=True 或 won=False）
2. 被截断（达到 max_steps 但未 done）

---

## 详细发现

### 1. Bad Cases 分析

检查 `bad_cases_step90_20260403_174747.json`：

| 指标 | 数值 |
|------|------|
| 验证集大小 | 1152 |
| Bad cases 数量 | 240 |
| 计算失败率 | 20.8% |
| 预估成功率 | ~79% |

**错误类型分布**：
| 错误类型 | 数量 |
|---------|------|
| action_not_available | 412 |
| abuse_safe_action | 119 |
| invalid | 29 |
| other | 276 |

### 2. 训练 vs Sandbox 配置差异

| 配置项 | 训练验证 | Sandbox |
|--------|----------|---------|
| `rollout.n` / `val_kwargs.n` | 1 | N/A |
| `temperature` | 0.4 | 0.4 |
| `do_sample` | True | True |
| `use_fallback_projection` | False | N/A |
| `max_steps` | 20 | 20 |
| available_skills 验证 | 启用 | 关闭（传入 None） |
| 解析失败处理 | 返回 INVALID_ACTION | fallback to gen_clarify |

**关键差异**：
- 训练时 `use_fallback_projection=False`，模型必须输出格式正确的 action
- Sandbox 调用 `customer_service_projection([output], None)` 跳过 available_skills 验证

### 3. 模型输出格式问题

分析 bad cases 发现大量 **reasoning_as_action**：

```
错误 action 示例：
"买家发送了商品链接，但没有明确表达具体咨询需求。根据规则，应优先选择澄清买家的需求"
```

这表明模型输出的 `<action>` 标签内包含 reasoning 而非 skill_id。

### 4. 环境容错机制

环境有 3 层容错：

```
Tier 1: 格式崩溃 (action ∉ VALID_SKILLS)
  → patience -= 1, reward -= 1.0

Tier 2a: 安全话术滥用 (action ∈ SAFE_FALLBACK_SKILLS, action ∉ transitions)
  → 前2次: reward -= 0.1
  → 3-4次: reward -= 0.3
  → 5次+: patience -= 1, reward -= 0.5

Tier 2b: 业务偏航 (action ∈ VALID_SKILLS, action ∉ transitions)
  → patience -= 1, reward -= 0.5

合法动作 (action ∈ transitions)
  → reward += 0.5, patience = min(2, patience + 1)
```

**模型有 2 次犯错机会**，但业务偏航会快速消耗耐心。

---

## 指标虚高的原因

### 原因 1：验证样本可能有重复成功

检查 `ray_trainer.py:710`：
```python
test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)
```

虽然 `val_kwargs.n=1`，但如果验证集有重复样本，可能导致同一个 playbook 被多次计入成功。

### 原因 2：短路径样本占比高

检查 bad cases 的 error_actions 分布：
- 1-2 errors: 110 cases (46%)
- 3-6 errors: 110 cases (46%)
- 7+ errors: 20 cases (8%)

短路径样本（rl_steps 少）更容易成功，可能导致整体指标偏高。

### 原因 3：训练时的 group sampling 偏差

训练时 `rollout.n=8`，每个样本跑 8 次。如果其中有 1-2 次成功，就会影响 GRPO 的 advantage 计算。

---

## 建议修复方案

### 1. 修复 success_rate 计算

```python
# etl/rl_interfaces.py - _process_batch
def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
    for i in reversed(range(len(total_batch_list[batch_idx]))):
        batch_item = total_batch_list[batch_idx][i]
        if batch_item.get('active_masks', False):
            info = total_infos[batch_idx][i]
            won_value = float(info.get('won', 0))
            success['success_rate'].append(won_value)

            # 新增：记录是否真正完成
            success['completed_rate'].append(1.0 if info.get('done', False) else 0.0)

            # 新增：记录截断率
            success['truncated_rate'].append(1.0 if not info.get('done', False) else 0.0)
            return
```

### 2. Sandbox 使用与训练一致的验证

```python
# viewer/services/sandbox_service.py - parse_model_output
def parse_model_output(output: str, available_skills: List[str] = None) -> Tuple[str, Optional[str]]:
    """使用与训练完全一致的解析逻辑"""
    from etl.rl_interfaces import customer_service_projection

    # 关键：传入 available_skills 进行完整验证
    actions, valids = customer_service_projection([output], [available_skills] if available_skills else None)

    action = actions[0] if actions else ""
    # ... 后续逻辑
```

### 3. 增加验证时的多路径评估

```bash
# 修改验证配置
actor_rollout_ref.rollout.val_kwargs.n=3  # 每个样本跑3次
actor_rollout_ref.rollout.val_kwargs.temperature=0.4
```

### 4. 分析 playbooks 的 rl_steps 分布

```python
# 检查是否有大量短路径样本
from collections import Counter
steps_dist = Counter(pb.get('rl_steps', 0) for pb in playbooks)
# 如果 rl_steps <= 5 的样本占比过高，可能导致指标虚高
```

---

## 下一步行动

1. **立即**：检查训练日志中的完整 episode 统计（done=True 的比例）
2. **短期**：修复 sandbox 的 available_skills 验证
3. **中期**：添加更详细的验证指标（completed_rate, truncated_rate）
4. **长期**：考虑增加验证集的多路径评估

---

## 附录：相关代码位置

| 文件 | 行号 | 功能 |
|------|------|------|
| `etl/customer_service_env.py` | 187-260 | 环境容错机制 |
| `etl/rl_interfaces.py` | 837-878 | success_evaluator |
| `verl/trainer/ppo/ray_trainer.py` | 706-810 | 验证循环 |
| `viewer/services/sandbox_service.py` | 195-219 | sandbox 解析 |
| `agent_system/multi_turn_rollout/rollout_loop.py` | 295-442 | rollout 循环 |