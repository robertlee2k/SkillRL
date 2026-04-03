# Bad Case Collection 代码机制说明

## 1. 数据流概览

```
Playbook数据
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ prepare_cs_data.py                                          │
│ - 读取 playbooks_all.json                                   │
│ - 每条playbook生成一条record                                 │
│ - record包含: prompt, data_source, extra_info               │
│ - extra_info = {playbook_id, scenario, session_id, ...}     │
│ - 输出 train.parquet / test.parquet                         │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ RLHFDataset.__getitem__()                                   │
│ - 从parquet读取record                                        │
│ - tokenizer处理prompt → input_ids, attention_mask           │
│ - extra_info保留在row_dict中                                 │
│ - 返回: {input_ids, attention_mask, data_source,            │
│         extra_info, ...}                                    │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ collate_fn()                                                 │
│ - 将多个sample组合成batch                                    │
│ - tensor数据: torch.stack → shape (batch_size, ...)         │
│ - 非tensor数据: np.array(dtype=object) → shape (batch_size,)│
│ - 返回: {input_ids: Tensor, data_source: np.array,          │
│         extra_info: np.array, ...}                          │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ DataProto.from_single_dict()                                │
│ - 将dict转换为DataProto对象                                  │
│ - tensor数据 → batch (TensorDict)                           │
│ - 非tensor数据 → non_tensor_batch (Dict[str, np.ndarray])   │
│ - extra_info在non_tensor_batch中                            │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ ray_trainer._validate() / training_loop                     │
│ - 从DataProto中pop env_kwargs (含playbook_id)               │
│ - 传递给 TrajectoryCollector.multi_turn_loop()              │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ rollout_loop.vanilla_multi_turn_loop()                      │
│ - envs.reset(kwargs=env_kwargs)                             │
│ - 多轮对话收集trajectory                                     │
│ - 返回: batch_list, rewards, success, traj_uid, ...         │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ ray_trainer._save_bad_cases()                               │
│ - 筛选score <= 0的失败样本                                   │
│ - 从env_kwargs提取playbook_id                               │
│ - 提取user_query, trajectory等信息                          │
│ - 输出JSON文件                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 关键数据结构

### 2.1 Playbook数据格式

```python
# outputs/playbooks_all.json 中的单条记录
{
    "playbook_id": "presale_001",
    "scenario": "presale",
    "subtype": "product_inquiry",
    "initial_buyer_message": "这个商品有货吗？",
    "available_skills": ["product_query", "inventory_check"],
    "dialogue_flow": [...],
    "rl_steps": 5,
    ...
}
```

### 2.2 训练数据格式 (prepare_cs_data.py 输出)

```python
# train.parquet / test.parquet 中的单条记录
{
    "data_source": "customer_service",
    "prompt": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "## 场景信息\n...## 买家消息\n这个商品有货吗？\n..."}
    ],
    "ability": "agent",
    "extra_info": {
        "playbook_id": "presale_001",
        "scenario": "presale",
        "subtype": "product_inquiry",
        "session_id": "xxx",
        "business_outcome": {...},
        "rl_steps": 5,
        "index": 0
    }
}
```

### 2.3 DataProto结构

```python
DataProto(
    batch=TensorDict({
        "input_ids": Tensor(batch_size, seq_len),
        "attention_mask": Tensor(batch_size, seq_len),
        "position_ids": Tensor(batch_size, seq_len),
    }),
    non_tensor_batch={
        "data_source": np.array(["customer_service", ...]),  # shape (batch_size,)
        "extra_info": np.array([{"playbook_id": "xxx", ...}, ...]),  # shape (batch_size,)
        "raw_prompt": np.array([...]),
        # ... 其他非tensor字段
    },
    meta_info={}
)
```

---

## 3. 关键代码路径

### 3.1 验证触发 (ray_trainer.py)

```python
# 第1596行
if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
   (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
    with _timer("testing", timing_raw):
        val_metrics: dict = self._validate()
```

配置参数：
- `test_freq=5`: 每5个训练step触发一次验证
- 验证时跑完**整个验证集**（1152样本）

### 3.2 验证循环 (ray_trainer.py 第689-793行)

```python
def _validate(self):
    # 初始化收集列表
    sample_inputs = []      # 解码后的完整prompt文本
    sample_outputs = []     # 解码后的模型输出
    sample_scores = []      # reward分数
    env_kwargs_list = []    # 环境参数(含playbook_id)
    data_source_lst = []    # 场景类型
    traj_uid_list = []      # 轨迹UUID

    # 遍历验证集的所有batch (9个batch, 每批128样本)
    for test_data in self.val_dataloader:
        test_batch = DataProto.from_single_dict(test_data)

        # 1. 解码输入文本
        input_ids = test_batch.batch["input_ids"]
        input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        sample_inputs.extend(input_texts)

        # 2. 准备生成batch，pop出env_kwargs
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
        if "env_kwargs" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("env_kwargs")
        test_gen_batch = test_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        # 3. 执行多轮rollout
        test_output_gen_batch = self.traj_collector.multi_turn_loop(
            gen_batch=test_gen_batch,
            actor_rollout_wg=self.actor_rollout_wg,
            envs=self.val_envs,
            is_train=False,  # 验证模式，不更新梯度
        )

        # 4. 解码输出
        output_ids = test_output_gen_batch.batch["responses"]
        output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        sample_outputs.extend(output_texts)

        # 5. 计算reward
        result = self.val_reward_fn(test_batch, return_dict=True)
        scores = result["reward_tensor"].sum(-1).cpu().tolist()
        sample_scores.extend(scores)

        # 6. 收集元数据
        env_kwargs_list.append(test_batch.non_tensor_batch.get('env_kwargs', [{}] * batch_size))
        data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * batch_size))
        traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])

    # 7. 拼接所有batch的数据
    env_kwargs = np.concatenate(env_kwargs_list, axis=0)
    data_sources = np.concatenate(data_source_lst, axis=0)
    traj_uids = np.concatenate(traj_uid_list, axis=0)

    # 8. 保存bad cases
    if self.config.trainer.get('save_bad_cases', False):
        self._save_bad_cases(
            sample_inputs=sample_inputs,
            sample_outputs=sample_outputs,
            sample_scores=sample_scores,
            success_rate=success_rate,
            data_sources=data_sources,
            traj_uids=traj_uids,
            env_kwargs=env_kwargs,
        )

    return metric_dict
```

### 3.3 Bad Case保存 (ray_trainer.py 第852-923行)

```python
def _save_bad_cases(
    self,
    sample_inputs: list,          # 解码后的完整prompt
    sample_outputs: list,         # 解码后的模型输出
    sample_scores: list,          # reward分数
    success_rate: dict,           # 成功率统计
    data_sources: 'np.ndarray | None' = None,  # 场景类型
    traj_uids: 'np.ndarray | None' = None,     # 轨迹UUID
    env_kwargs: 'np.ndarray | None' = None,    # 环境参数
):
    bad_cases = []

    for i, (inp, out, score) in enumerate(zip(sample_inputs, sample_outputs, sample_scores)):
        if score <= 0:  # 失败轨迹
            # 1. 提取唯一标识符
            playbook_id = ''
            scenario = ''
            if env_kwargs is not None and i < len(env_kwargs):
                env_kw = env_kwargs[i]
                if isinstance(env_kw, dict):
                    playbook_id = env_kw.get('playbook_id', '')
            if data_sources is not None and i < len(data_sources):
                scenario = str(data_sources[i])

            # 2. 提取用户查询
            user_query = self._extract_user_query(inp)

            # 3. 解析轨迹
            trajectory = self._parse_conversation_to_steps(inp, out)

            bad_cases.append({
                'playbook_id': playbook_id,   # 唯一标识符，用于去重
                'scenario': scenario,
                'user_query': user_query,
                'trajectory': trajectory,
                'score': score,
            })

    # 4. 保存JSON
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'global_steps': self.global_steps,
        'total_samples': len(sample_inputs),
        'total_failures': len(bad_cases),
        'success_rate': success_rate,
        'bad_cases': bad_cases,
    }

    filename = os.path.join(
        output_dir,
        f'bad_cases_step{self.global_steps}_{datetime.now():%Y%m%d_%H%M%S}.json'
    )
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
```

### 3.4 Rollout循环 (rollout_loop.py 第309行)

```python
def vanilla_multi_turn_loop(self, gen_batch, actor_rollout_wg, envs):
    batch_size = len(gen_batch.batch)

    # 从gen_batch中获取env_kwargs，传递给环境reset
    obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

    # 生成轨迹UID (每次rollout随机生成)
    traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)

    # 多轮对话循环...
    for _step in range(self.config.env.max_steps):
        # 生成action
        batch_output = actor_rollout_wg.generate_sequences(batch_input)
        # 执行action
        next_obs, rewards, dones, infos = envs.step(text_actions)
        # ...

    return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
```

---

## 4. env_kwargs的数据来源

### 4.1 问题：env_kwargs从哪来？

**答案**：`extra_info` 字段在数据处理时被重命名为 `env_kwargs`。

### 4.2 数据流追踪

1. **数据准备阶段** (`prepare_cs_data.py`):
   ```python
   record = {
       'data_source': 'customer_service',
       'prompt': [...],
       'ability': 'agent',
       'extra_info': {
           'playbook_id': playbook_id,
           'scenario': scenario,
           ...
       }
   }
   ```

2. **Dataset加载阶段** (`rl_dataset.py`):
   - `RLHFDataset.__getitem__()` 返回的 `row_dict` 包含 `extra_info`

3. **DataProto创建阶段**:
   - `from_single_dict()` 将 `extra_info` 放入 `non_tensor_batch`

4. **使用阶段**:
   - 如果环境需要，`extra_info` 会被当作 `env_kwargs` 传递给 `envs.reset()`
   - CustomerService环境使用 `playbook_id` 来选择具体的对话场景

### 4.3 实际验证

检查 `non_tensor_batch` 中是否有 `env_kwargs`:
```python
# ray_trainer.py 第726-727行
if "env_kwargs" in test_batch.non_tensor_batch:
    non_tensor_batch_keys_to_pop.append("env_kwargs")
```

如果 `env_kwargs` 存在，则会被pop出来传递给环境。

---

## 5. traj_uid的作用

### 5.1 生成位置
```python
# rollout_loop.py 第325行
traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
```

### 5.2 作用范围
- **仅在一次rollout内有效**
- 用于区分同一个batch内的不同轨迹
- 用于计算GRPO的group内优势值

### 5.3 不能用于去重
- 每次rollout都会生成新的UUID
- 同一个 `playbook_id` 在不同验证周期会有不同的 `traj_uid`
- **去重应该用 `playbook_id`**

---

## 6. 用户查询提取

### 6.1 Prompt格式
```
<|im_start|>system
你是一名专业的电商客服智能助手...
<|im_end|>
<|im_start|>user
## 场景信息
场景类型: 售前咨询 (presale)
子类型: product_inquiry

## 买家消息
这个商品有货吗？

## 任务
请分析买家的需求，并选择合适的客服动作进行回应...
<|im_end|>
```

### 6.2 提取方法
```python
def _extract_user_query(self, inp: str) -> str:
    import re

    # ChatML格式
    user_match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', inp, re.DOTALL)
    if user_match:
        user_content = user_match.group(1).strip()
        # 提取买家消息
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

## 7. 配置参数

### 7.1 启用Bad Case收集

```bash
# run_customer_service.sh
python3 -m verl.trainer.main_ppo \
    ... \
    +trainer.save_bad_cases=True \
    +trainer.bad_cases_output_dir=outputs/bad_cases \
    ...
```

### 7.2 相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `trainer.test_freq` | 5 | 每N个step触发验证 |
| `data.val_batch_size` | 128 | 验证批次大小 |
| `env.max_steps` | 20 | 最大对话轮数 |
| `trainer.save_bad_cases` | False | 是否保存失败案例 |
| `trainer.bad_cases_output_dir` | outputs/bad_cases | 输出目录 |

---

## 8. 输出文件格式

```json
{
  "timestamp": "2025-03-29T15:30:00",
  "global_steps": 85,
  "total_samples": 1152,
  "total_failures": 380,
  "success_rate": {
    "presale_success_rate": 0.72,
    "aftersale_success_rate": 0.65,
    "logistics_success_rate": 0.70
  },
  "bad_cases": [
    {
      "playbook_id": "presale_001",
      "scenario": "presale",
      "user_query": "这个商品有货吗？",
      "trajectory": [
        {"action": "inventory_check", "observation": "..."},
        {"action": "gen_clarify", "observation": "..."},
        ...
      ],
      "score": -0.5
    },
    ...
  ]
}
```

---

## 9. 去重策略

```python
import json
from collections import defaultdict

# 加载所有bad case文件
all_bad_cases = []
for file in glob.glob("outputs/bad_cases/*.json"):
    with open(file, 'r') as f:
        data = json.load(f)
        all_bad_cases.extend(data['bad_cases'])

# 按playbook_id去重
unique_bad_cases = {}
for case in all_bad_cases:
    playbook_id = case['playbook_id']
    if playbook_id not in unique_bad_cases:
        unique_bad_cases[playbook_id] = case
    # 如果已存在，可以选择保留score更低的（失败更严重的）
    elif case['score'] < unique_bad_cases[playbook_id]['score']:
        unique_bad_cases[playbook_id] = case

print(f"Total: {len(all_bad_cases)}, Unique: {len(unique_bad_cases)}")
```

---

## 10. 现有问题总结

| 问题 | 原因 | 影响 |
|------|------|------|
| 所有bad case的task字段相同 | `_extract_task_description`返回系统prompt开头 | 无法区分不同任务 |
| 缺少playbook_id | 原代码没有从env_kwargs提取 | 无法去重 |
| 买家消息提取错误 | 正则表达式格式不匹配 | 数据不准确 |
| traj_uid误用 | 以为是持久ID | 无法跨周期去重 |

以上问题已在新代码中修复。现有6561条数据因缺少 `playbook_id`，无法可靠去重或修复。