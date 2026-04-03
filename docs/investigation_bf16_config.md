# veRL 强化学习训练配置调查报告

## 背景

使用 8 张 80G A100 显卡，通过 GRPO 算法微调 Qwen 2.5 14B 模型。初始权重是 FP32 格式，需要确保整个训练过程（Actor 训练、Reference/Reward 推理）都在 BF16 精度下进行，以避免 OOM 和数值溢出。

---

## 1. 模型加载精度配置

### 关键文件

`verl/workers/fsdp_workers.py`

### 问题发现

**Actor 模型加载**（第 203-207 行）:
```python
torch_dtype = fsdp_config.get("model_dtype", None)
if torch_dtype is None:
    torch_dtype = torch.float32 if self._is_actor else torch.bfloat16  # ⚠️ 问题在这里！
else:
    torch_dtype = PrecisionType.to_dtype(torch_dtype)
```

**问题**: Actor 模型默认加载为 **FP32**，而 Reference 模型默认加载为 **BF16**。这是一个不一致且可能导致 OOM 的问题。

**Critic 模型加载**（第 874-876 行）:
```python
torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")  # ⚠️ 默认 FP32！
torch_dtype = PrecisionType.to_dtype(torch_dtype)
```

**Reward Model 加载**（第 1199-1204 行）:
```python
reward_module = AutoModelForTokenClassification.from_pretrained(
    ...
    torch_dtype=torch.bfloat16,  # ✅ 硬编码为 BF16
    ...
)
```

### FSDP Mixed Precision 配置

（第 290-300 行）:
```python
mixed_precision_config = fsdp_config.get("mixed_precision", None)
if mixed_precision_config is not None:
    param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
    reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
    buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
else:
    param_dtype = torch.bfloat16  # 默认是 BF16
    reduce_dtype = torch.float32
    buffer_dtype = torch.float32
```

**结论**: FSDP 的 mixed_precision 默认使用 BF16，但模型加载时的 `torch_dtype` 是独立的设置，不受 mixed_precision 影响。

---

## 2. DeepSpeed 配置

### 发现

该仓库**不使用 DeepSpeed**，而是使用 PyTorch 原生的 **FSDP (Fully Sharded Data Parallel)**。

### FSDP Sharding Strategy

（第 91-97 行）:
```python
def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy
    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD  # 等同于 ZeRO-3
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD  # ZeRO-3 + DDP
```

**结论**: FSDP 使用 `FULL_SHARD` 策略，等同于 DeepSpeed ZeRO-3，适合大模型训练。

---

## 3. vLLM 显存分配配置

### 关键文件

`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

### 配置代码

（第 185-207 行）:
```python
self.inference_engine = LLM(
    ...
    dtype=config.dtype,                       # 使用配置中的 dtype
    gpu_memory_utilization=config.gpu_memory_utilization,  # 使用配置中的值
    ...
)
```

### YAML 默认配置

`verl/trainer/config/ppo_trainer.yaml` 第 114-115 行:
```yaml
rollout:
  dtype: bfloat16                    # ✅ 正确配置为 BF16
  gpu_memory_utilization: 0.5        # 当前设置为 0.5
```

### 显存评估

对于 14B 模型：
- BF16 参数大小: ~28GB
- 建议的 `gpu_memory_utilization`: 0.3-0.4（避免与训练显存冲突）

---

## 4. 需要修改的文件和具体内容

### 文件 1: `verl/workers/fsdp_workers.py`

#### 修改位置 1 - Actor 模型加载（第 203-207 行）

```python
# 原代码
torch_dtype = fsdp_config.get("model_dtype", None)
if torch_dtype is None:
    torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
else:
    torch_dtype = PrecisionType.to_dtype(torch_dtype)

# 建议修改为
torch_dtype = fsdp_config.get("model_dtype", None)
if torch_dtype is None:
    torch_dtype = torch.bfloat16  # 统一使用 BF16
else:
    torch_dtype = PrecisionType.to_dtype(torch_dtype)
```

#### 修改位置 2 - Critic 模型加载（第 874-876 行）

```python
# 原代码
torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
torch_dtype = PrecisionType.to_dtype(torch_dtype)

# 建议修改为
torch_dtype = self.config.model.fsdp_config.get("model_dtype", "bf16")
torch_dtype = PrecisionType.to_dtype(torch_dtype)
```

---

### 文件 2: `verl/trainer/config/ppo_trainer.yaml`

在 fsdp_config 下添加 `model_dtype: bf16`:

```yaml
# Actor 配置
actor:
  fsdp_config:
    model_dtype: bf16    # 新增：显式指定模型加载精度为 BF16
    wrap_policy:
      min_num_params: 0
    param_offload: False
    optimizer_offload: False
    ...

# Reference 配置
ref:
  fsdp_config:
    model_dtype: bf16    # 新增
    param_offload: False
    ...

# Critic 配置
critic:
  model:
    fsdp_config:
      model_dtype: bf16  # 新增
      ...
```

---

### 文件 3: 启动脚本参数

在命令行添加：

```bash
python3 -m verl.trainer.main_ppo \
    ...
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    critic.model.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    ...
```

---

## 5. 总结

| 组件 | 当前状态 | 建议 |
|------|---------|------|
| Actor 模型加载 | 默认 FP32 ❌ | 改为 BF16 |
| Reference 模型加载 | 默认 BF16 ✅ | 保持 BF16 |
| Critic 模型加载 | 默认 FP32 ❌ | 改为 BF16 |
| Reward Model 加载 | 硬编码 BF16 ✅ | 保持不变 |
| vLLM dtype | BF16 ✅ | 保持不变 |
| vLLM gpu_memory_utilization | 0.5 | 对于 14B 建议降到 0.3-0.4 |
| DeepSpeed | 未使用 | 使用 FSDP FULL_SHARD（等同于 ZeRO-3）✅ |

---

## 6. 显存估算

### 7B 模型（BF16）

- 模型参数: ~14GB
- 梯度: ~14GB（FSDP 分片后每 GPU ~1.75GB，8 GPU）
- 优化器状态: ~28GB（AdamW，分片后每 GPU ~3.5GB）
- 激活值: 取决于 batch size 和 sequence length

### 14B 模型（BF16）

- 模型参数: ~28GB
- 梯度: ~28GB（FSDP 分片后每 GPU ~3.5GB，8 GPU）
- 优化器状态: ~56GB（AdamW，分片后每 GPU ~7GB）
- 激活值: 取决于 batch size 和 sequence length

### 关键问题

Actor 和 Critic 模型默认加载为 FP32 会导致：
1. 显存占用翻倍（FP32 vs BF16）
2. 可能导致 OOM
3. 训练效率降低

**最关键的修复**: 确保 `model_dtype` 配置为 `bf16`。

---

## 7. 实际显存使用分析（Qwen2.5-7B-Instruct）

### 训练配置

```yaml
# 实际运行的配置
actor_rollout_ref:
  model:
    path: /home/bo.li/data/models/Qwen2.5-7B-Instruct
  actor:
    fsdp_config:
      param_offload: True      # 参数卸载到CPU
      optimizer_offload: True  # 优化器卸载到CPU
  rollout:
    gpu_memory_utilization: 0.5
    tensor_model_parallel_size: 4
    dtype: bfloat16
```

### 实测显存数据

| 指标 | 数值 |
|------|------|
| 峰值显存分配 (max_memory_allocated) | **58.54 GB** |
| 峰值显存保留 (max_memory_reserved) | **85.35 GB** |
| CPU 内存使用 | 517.84 GB |
| MFU (Model FLOPs Utilization) | 30.17% |

### 当前显存使用（训练进行中）

| GPU | 使用量 | 总量 | 使用率 |
|-----|--------|------|--------|
| GPU 0 | 44.78 GB | 80 GB | 55% |
| GPU 1 | 43.36 GB | 80 GB | 53% |
| GPU 2 | 49.27 GB | 80 GB | 60% |
| GPU 3 | 42.85 GB | 80 GB | 52% |
| GPU 4 | 44.17 GB | 80 GB | 54% |
| GPU 5 | 42.15 GB | 80 GB | 51% |
| GPU 6 | 47.28 GB | 80 GB | 58% |
| GPU 7 | 49.61 GB | 80 GB | 60% |

### vLLM 推理显存

从 Ray worker 日志可以看到：
```
Sleep mode freed 37.61 GiB memory, 7.20 GiB memory is still in use.
```

- **vLLM 活跃时显存**: ~45 GiB (37.61 + 7.20)
- **vLLM Sleep 时保留显存**: ~7.20 GiB

### 显存分析与估算对比

**7B 模型 BF16 理论计算（不开启 offload）**:
- 模型参数: ~14 GB
- 优化器状态 (AdamW, FP32 主权重 + 动量): ~28 GB
- 梯度: ~14 GB
- **总计**: ~56 GB (单 GPU 不分片)

**使用 FSDP ZeRO-3 + 8 GPU 分片**:
- 每GPU参数分片: 56 / 8 = 7 GB

**实际观察**:
- 由于开启了 `param_offload=True` 和 `optimizer_offload=True`
- 参数和优化器状态卸载到 CPU
- GPU 显存主要用于：激活值、临时计算缓冲区、vLLM 推理

### 14B 模型预估

如果将 7B 模型换成 14B 模型：

| 组件 | 7B (BF16) | 14B (BF16) |
|------|-----------|------------|
| 模型参数 | ~14 GB | ~28 GB |
| 优化器状态 | ~28 GB | ~56 GB |
| 梯度 | ~14 GB | ~28 GB |
| **总计（不分片）** | ~56 GB | ~112 GB |
| **每 GPU (ZeRO-3, 8卡)** | ~7 GB | ~14 GB |

**预估峰值显存（开启 offload）**:
- 7B: ~58 GB (实测)
- 14B: ~75-85 GB (预估，需要关闭 offload 或使用更多显存优化)

### 建议

对于 14B 模型：
1. 保持 `param_offload=True` 和 `optimizer_offload=True`
2. 降低 `gpu_memory_utilization` 到 0.3-0.35
3. 减小 batch size 或使用 gradient accumulation
4. 考虑使用 `enable_gradient_checkpointing=True` (已开启)

---

## 9. 显存分析结论

### 7B 模型实际显存使用

| 指标 | 数值 |
|------|------|
| 峰值显存分配 | **58.54 GB** |
| 峰值显存保留 | **85.35 GB** |
| 当前使用（各GPU） | 42-50 GB / 80 GB |

### 是否符合估算？

**符合预期**，分析如下：

1. **当前配置开启了 offload**:
   - `param_offload=True`: 参数卸载到 CPU
   - `optimizer_offload=True`: 优化器卸载到 CPU
   - 这大幅降低了 GPU 显存占用

2. **vLLM 推理显存**:
   - Sleep 模式释放: 37.61 GiB
   - 基础保留: 7.20 GiB
   - 活跃时总计: ~45 GiB

3. **估算对比**:
   - 理论值（不 offload）: 7B 模型需要 ~56 GB 参数+优化器+梯度
   - 实测值（开启 offload）: 峰值 58.54 GB（主要是激活值 + vLLM）
   - **结论**: 因为 offload，GPU 显存主要用于激活值和推理，与理论一致

### 14B 模型预估

如果换成 14B 模型：
- 参数量翻倍 → 显存需求增加约 30-40%
- 预估峰值: **75-85 GB**（开启 offload）
- 建议降低 `gpu_memory_utilization` 到 0.3-0.35

---

## 10. 相关文件路径

- 主配置文件: `verl/trainer/config/ppo_trainer.yaml`
- FSDP Workers: `verl/workers/fsdp_workers.py`
- vLLM Rollout: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- 精度类型工具: `verl/utils/torch_dtypes.py`
- 示例启动脚本: `examples/grpo_trainer/run_alfworld.sh`