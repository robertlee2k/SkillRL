# Bad Case Collection 问题分析与修复报告

## 背景

在CustomerService强化学习训练过程中，我们通过 `+trainer.save_bad_cases=True` 配置启用了失败案例收集功能，用于后续的skill evolution分析。训练运行几个小时后，发现收集了**6561条bad cases**，但存在严重的数据质量问题。

---

## 问题1：为什么有6561条bad cases？

### 原始理解误区

最初以为每次validation只跑 `val_batch_size=128` 个样本，因此6561条数据看起来异常。

### 实际机制

| 阶段 | 配置参数 | 实际处理量 |
|------|----------|-----------|
| **训练** | `train_batch_size=64` | 每个step处理64个样本 |
| **验证** | `val_batch_size=128` | 每次validation跑**整个验证集**（1152样本） |

验证集大小：1152样本（由数据预处理脚本 `prepare_cs_data.py` 生成）
验证批次划分：1152 ÷ 128 = **9个批次**

代码证据（`ray_trainer.py`）：
```python
# 第605-621行：创建dataloader
val_batch_size = self.config.data.val_batch_size  # 128
self.val_dataloader = StatefulDataLoader(
    dataset=self.val_dataset,
    batch_size=val_batch_size,  # 128
)
print(f"Size of val dataloader: {len(self.val_dataloader)}")  # 输出: 9

# 第702行：验证循环
for test_data in self.val_dataloader:  # 循环9次
    test_batch = DataProto.from_single_dict(test_data)
    # 每次处理128个，但循环跑完整个验证集
```

验证触发频率：`test_freq=5`，即每5个训练step触发一次完整验证。

### 数量计算

假设训练运行到step 85：
- 验证次数：85 ÷ 5 = 17次
- 每次验证：1152样本
- 总验证样本：17 × 1152 = 19,584次轨迹采样

假设失败率约33%（成功率67%）：
- 失败样本：19,584 × 33% ≈ 6,500

**结论：6561条bad cases数量正常**，因为每次validation都跑完整验证集。

---

## 问题2：验证集是否参与梯度更新？

**不会。验证阶段只做评估，不更新梯度。**

代码证据：
```python
# 验证阶段（第754行）
test_output_gen_batch = self.traj_collector.multi_turn_loop(
    gen_batch=test_gen_batch,
    actor_rollout_wg=self.actor_rollout_wg,
    envs=self.val_envs,
    is_train=False,  # <-- 不训练
)
# 后续只计算reward和metrics，返回，没有backward

# 训练阶段（第1411行）
gen_batch_output = self.traj_collector.multi_turn_loop(
    gen_batch=gen_batch,
    actor_rollout_wg=self.actor_rollout_wg,
    envs=self.envs,
    is_train=True,  # <-- 用于训练
)
# 后续有：compute_advantage、backward、optimizer.step()
```

---

## 问题3：现有数据的致命缺陷

### Bug描述

所有6561条bad cases的 `task` 字段**完全相同**，都是系统prompt的开头部分：

```
system\n你是一名专业的电商客服智能助手，擅长处理售前咨询、物流查询和售后问题...
```

### 原因分析

原始 `_save_bad_cases()` 方法：
```python
def _save_bad_cases(self, sample_inputs, sample_outputs, sample_scores, success_rate):
    for inp, out, score in zip(sample_inputs, sample_outputs, sample_scores):
        if score <= 0:
            task_desc = self._extract_task_description(inp)  # BUG: 返回系统prompt
            bad_cases.append({'task': task_desc, ...})
```

`_extract_task_description()` 方法的问题：
```python
def _extract_task_description(self, inp: str) -> str:
    # Fallback逻辑：找第一个user turn
    for marker in ('<|im_start|>user\n', '\nHuman: ', '\nUser: '):
        idx = inp.find(marker)
        if idx >= 0:
            start = idx + len(marker)
            return inp[start:start + 1000]  # 返回整个user内容，太长且不精确
    return inp[:1000]  # 兜底返回开头（系统prompt）
```

### 缺失的唯一标识符

原始代码没有保存：
- `playbook_id`：每个playbook的唯一标识符
- `scenario`：场景类型（presale/aftersale/logistics）
- `traj_uid`：轨迹唯一ID

**后果**：无法对6561条数据进行去重，不知道哪些是同一个playbook重复失败的。

---

## 问题4：买家消息提取错误

### 我的修复Bug

在修复 `_extract_user_query()` 时，我最初写了一个错误的正则：

```python
# 错误版本
buyer_match = re.search(r'买家消息[：:]\s*(.*?)(?:\n|$)', user_content, re.DOTALL)
```

这个正则匹配的是 `买家消息：xxx` 或 `买家消息:xxx` 格式。

### 实际Prompt格式

查看 `prepare_cs_data.py` 第409-420行：

```python
prompt = f"""## 场景信息
场景类型: {scenario_desc} ({scenario})
子类型: {subtype}

## 买家消息
{buyer_text}

## 任务
请分析买家的需求...
"""
```

实际格式是：
```
## 买家消息
{buyer_text}    # <-- 这里是换行，不是冒号
```

### 修正后的正则

```python
# 正确版本
buyer_match = re.search(
    r'## 买家消息\s*\n(.*?)(?=\n##|\n<\|im_end\|>|$)',
    user_content, re.DOTALL
)
```

---

## 问题5：traj_uid的作用

### traj_uid是什么？

代码证据（`rollout_loop.py` 第325行）：
```python
traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
```

**每次rollout开始时随机生成的UUID**，用于区分同一批次内的不同轨迹。

### traj_uid对去重有用吗？

**没用。**

- 同一个 `playbook_id`，每次validation都会生成新的 `traj_uid`
- `traj_uid` 是一次rollout内的临时标识，不是跨周期的持久标识
- 真正用于去重应该用 `playbook_id`

### 去重策略

同一个 `playbook_id` 多次失败（出现在多个验证周期），只保留一条记录即可。

---

## 已完成的修复

### 修改文件：`verl/trainer/ppo/ray_trainer.py`

#### 1. 收集env_kwargs（第693-776行）

```python
env_kwargs_list = []  # NEW: collect env_kwargs for playbook_id

# 在validation循环中收集
env_kwargs_list.append(test_batch.non_tensor_batch.get('env_kwargs', [{}] * reward_tensor.shape[0]))

# 拼接
env_kwargs = np.concatenate(env_kwargs_list, axis=0)
```

#### 2. 传递参数给 `_save_bad_cases`（第830-838行）

```python
self._save_bad_cases(
    sample_inputs=sample_inputs,
    sample_outputs=sample_outputs,
    sample_scores=sample_scores,
    success_rate=success_rate,
    data_sources=data_sources,       # NEW
    env_kwargs=env_kwargs,           # NEW
)
```

#### 3. 修复 `_detect_task_type_from_input` 硬编码问题

**问题**：原方法硬编码返回 `'pick_and_place'`，这是 AlfWorld 的任务类型，对 CustomerService 无意义。

**修复**：添加 CustomerService scenario 提取逻辑

```python
def _detect_task_type_from_input(self, inp: str) -> str:
    """从输入中检测任务类型

    For CustomerService: extracts scenario (presale/aftersale/logistics)
    For AlfWorld: extracts task type (pick_and_place/heat/cool/clean/examine/look_at_obj_in_light)
    """
    # CustomerService: 场景类型: 售后服务 (aftersale)
    scenario_match = inp and re.search(
        r'场景类型:\s*[^\n\(]*\((\w+)\)',
        inp
    )
    if scenario_match:
        return scenario_match.group(1)  # presale/aftersale/logistics

    # AlfWorld task types (fallback)
    inp_lower = inp.lower() if inp else ''
    if 'clean' in inp_lower:
        return 'clean'
    elif 'heat' in inp_lower:
        return 'heat'
    # ... 其他 AlfWorld 任务类型
    else:
        return 'unknown'
```

#### 4. 简化 bad case 字段

移除了冗余的 `task_type` 和 `traj_uid` 字段，保留核心字段：

```python
bad_cases.append({
    'playbook_id': playbook_id,    # 唯一标识符
    'scenario': scenario,          # presale/aftersale/logistics
    'user_query': user_query,      # 买家消息
    'trajectory': trajectory,      # 完整轨迹
    'score': score,
})
```

#### 3. 重写 `_save_bad_cases` 方法

```python
def _save_bad_cases(
    self,
    sample_inputs: list,
    sample_outputs: list,
    sample_scores: list,
    success_rate: dict,
    data_sources: 'np.ndarray | None' = None,
    traj_uids: 'np.ndarray | None' = None,
    env_kwargs: 'np.ndarray | None' = None,
):
    bad_cases = []
    for i, (inp, out, score) in enumerate(zip(sample_inputs, sample_outputs, sample_scores)):
        if score <= 0:
            # 获取唯一标识符
            playbook_id = ''
            scenario = ''
            if env_kwargs is not None and i < len(env_kwargs):
                env_kw = env_kwargs[i]
                if isinstance(env_kw, dict):
                    playbook_id = env_kw.get('playbook_id', '')
            if data_sources is not None and i < len(data_sources):
                scenario = str(data_sources[i])

            user_query = self._extract_user_query(inp)  # 新方法
            trajectory = self._parse_conversation_to_steps(inp, out)

            bad_cases.append({
                'playbook_id': playbook_id,  # 关键：唯一标识符
                'scenario': scenario,
                'user_query': user_query,
                'trajectory': trajectory,
                'score': score,
            })
```

#### 4. 新增 `_extract_user_query` 方法

```python
def _extract_user_query(self, inp: str) -> str:
    """提取买家消息"""
    import re

    # ChatML格式
    user_match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', inp, re.DOTALL)
    if user_match:
        user_content = user_match.group(1).strip()
        # 正确的正则
        buyer_match = re.search(
            r'## 买家消息\s*\n(.*?)(?=\n##|\n<\|im_end\|>|$)',
            user_content, re.DOTALL
        )
        if buyer_match:
            return buyer_match.group(1).strip()[:500]
        return user_content[:500]

    return inp[:500]
```

---

## 现有数据的状态

### 数据文件位置

```
outputs/bad_cases/
├── bad_cases_step0_20250329_xxx.json
├── bad_cases_step5_20250329_xxx.json
├── bad_cases_step10_20250329_xxx.json
├── ...
└── bad_cases_step85_20250329_xxx.json  # 共18个文件
```

### 数据缺陷

| 字段 | 现状 | 应有内容 |
|------|------|----------|
| `task` | ❌ 全部相同（系统prompt开头） | 应为 `playbook_id` |
| `playbook_id` | ❌ 缺失 | 应为唯一标识符 |
| `scenario` | ❌ 缺失 | 应为场景类型 |
| `user_query` | ❌ 缺失 | 应为买家消息 |
| `trajectory` | ✓ 有 | 正确 |
| `score` | ✓ 有 | 正确 |

### 无法修复的原因

1. `playbook_id` 没有保存，无法追溯
2. 原始prompt已解码，无法重新提取买家消息（因为解码后的格式是ChatML，需要知道具体格式才能准确提取）
3. 无法区分哪些是同一个playbook重复失败

---

## 建议方案（待Gemini审核）

### 方案A：丢弃现有数据，重新收集

- 等待当前训练完成，或重启训练
- 新的bad case文件会有正确的 `playbook_id` 和 `user_query`
- 可以进行去重分析

**优点**：数据质量可靠
**缺点**：丢失已收集的6561条数据

### 方案B：尝试部分修复现有数据

- 从 `trajectory` 中提取场景信息（可能不准确）
- 手动关联到原始 `playbooks_all.json`（困难，无唯一标识符）
- 推测去重（不可靠）

**优点**：保留数据量
**缺点**：数据质量存疑，分析结论可能误导

### 方案C：混合方案

1. 尝试从现有数据的 `trajectory` 字段提取有限信息（如失败action类型）
2. 记录分析结论的置信度（低）
3. 等新数据收集后，进行对比验证

---

## 附录：数据流示意图

```
训练循环
│
│  每5个step触发验证 (test_freq=5)
│
├──► validation开始
│    │
│    │  验证集 = 1152样本
│    │  分9批处理 (每批128)
│    │
│    ├── Batch 1: 128样本 → rollout → reward计算
│    ├── Batch 2: 128样本 → rollout → reward计算
│    ├── ...
│    ├── Batch 9: 128样本 → rollout → reward计算
│    │
│    │  收集所有结果 (1152个)
│    │  失败样本 → _save_bad_cases()
│    │
│    └──► 输出: bad_cases_step{N}.json
│
│  返回metrics，不更新梯度
│
└──► 继续训练
```

---

## 相关文件

| 文件 | 作用 |
|------|------|
| `verl/trainer/ppo/ray_trainer.py` | 主训练逻辑，包含 `_save_bad_cases` |
| `scripts/prepare_cs_data.py` | 数据预处理，定义prompt格式 |
| `agent_system/multi_turn_rollout/rollout_loop.py` | 多turn rollout逻辑，生成 `traj_uid` |
| `examples/grpo_trainer/run_customer_service.sh` | 训练启动脚本 |