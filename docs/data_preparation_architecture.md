# RL 训练数据准备脚本架构文档

## 概述

`scripts/prepare_cs_data.py` 是一个工业级的数据准备引擎，用于将 Playbook JSON 数据转换为 veRL/GiGPO 训练所需的 Parquet 格式。

## 核心特性

### 1. O(1) RL Steps 过滤

利用新增的 `rl_steps` 字段，在 O(1) 复杂度下完成过滤：

```python
# 每个 playbook 只需一次比较操作
if rl_steps is None or rl_steps <= max_rl_steps:
    retained.append(p)
else:
    discarded.append(p)
```

**优势：**
- 无需重新计算 turn count
- 保守策略：缺失 `rl_steps` 的数据被保留（避免误删）
- 实时统计被丢弃数据的分布

### 2. 分层抽样 (Stratified Sampling)

按 `scenario` 字段进行分层抽样，确保 train/val 分布一致：

```python
# 按 scenario 分组
groups = {'presale': [...], 'aftersale': [...], 'logistics': [...], 'unknown': [...]}

# 每组内独立按 val_ratio 分割
for key, group in groups:
    val_size = int(len(group) * val_ratio)
    # 组内 shuffle 后分割
```

**分布一致性验证：**
- Train/Val 相对误差 < 0.5%
- 自动计算并报告分布匹配度

### 3. 自动 Drop-Last 对齐

确保数据集大小被 batch_size 整除，避免训练时的 batch size mismatch 错误：

```
原始 Train: 4918 → 对齐后: 4864 (76 batches × 64)
原始 Val:   865  → 对齐后: 768  (6 batches × 128)
```

**历史问题回顾：**
- 之前 val_data_size=1200，最后 batch 只有 48 样本
- 导致 `gen_batch size 48 does not match obs size 128` 错误

### 4. Fail-Fast 断言

在关键节点设置断言，确保数据质量：

```python
# Stage 2: 过滤后数据量检查
if len(filtered) < train_batch_size + val_batch_size:
    raise ValueError("Insufficient data after filtering")

# Stage 3: 分割后最小批次检查
if len(train_playbooks) < train_batch_size:
    raise ValueError("Train split smaller than batch size")

# Stage 5: 输出验证
- Batch alignment check
- Required columns check
- Prompt format check
- Duplicate playbook_id check
- RL steps threshold check
```

### 5. Rich Logging & Report

生成 `report.json` 包含完整审计追踪：

```json
{
  "stage_1_input": { "total": 5784, "scenario_distribution": {...} },
  "stage_2_rl_steps_filter": { "retention_rate": "99.98%", "discarded": 1 },
  "stage_3_stratified_split": { "distribution_match": {"max_train_error": 0.03%} },
  "stage_4_drop_last_alignment": { "train_batches": 76, "val_batches": 6 },
  "stage_5_output": { "verification_passed": true }
}
```

## 参数设计

### Ratio-Driven 参数（推荐）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--val_ratio` | 0.15 | 验证集比例 |
| `--train_batch_size` | 64 | 训练批次大小（用于对齐） |
| `--val_batch_size` | 128 | 验证批次大小（用于对齐） |
| `--stratify_by` | scenario | 分层抽样字段 |
| `--max_rl_steps` | 20 | RL steps 上限阈值 |

### 废弃参数（不再使用）

| 参数 | 替代方案 |
|------|----------|
| `--train_data_size` | 由 `total × (1 - val_ratio)` 自动计算 |
| `--val_data_size` | 由 `total × val_ratio` 自动计算 |

## 使用示例

```bash
# 默认用法
python scripts/prepare_cs_data.py

# 自定义参数
python scripts/prepare_cs_data.py \
    --playbook_path outputs/playbooks_all.json \
    --output_dir ~/data/verl-agent/customer_service \
    --val_ratio 0.15 \
    --train_batch_size 64 \
    --val_batch_size 128 \
    --max_rl_steps 20

# 更严格 RL steps 限制
python scripts/prepare_cs_data.py --max_rl_steps 15

# 更大验证集
python scripts/prepare_cs_data.py --val_ratio 0.2
```

## 输出文件

```
output_dir/
├── train.parquet   # 训练数据（对齐到 train_batch_size）
├── test.parquet    # 验证数据（对齐到 val_batch_size）
└── report.json     # 完整审计报告
```

## Pipeline 流程图

```
Stage 1: Load Input
    └── Load playbooks JSON
    └── Compute scenario distribution
    └── Compute RL steps stats

Stage 2: RL Steps Filter (O(1))
    └── Discard if rl_steps > max_rl_steps
    └── Log retention rate & discarded distribution

Stage 3: Stratified Split
    └── Group by stratify_by field
    └── Split each group by val_ratio
    └── Verify distribution match

Stage 4: Drop-Last Alignment
    └── Align train to train_batch_size
    └── Align val to val_batch_size

Stage 5: Output & Verify
    └── Convert to records
    └── Run fail-fast verification
    └── Save parquet files
    └── Generate report.json
```

## 关键指标示例

基于 `playbooks_all.json` (5784 条) 的运行结果：

| 指标 | 值 |
|------|-----|
| 输入总量 | 5784 |
| RL Steps 过滤保留率 | 99.98% (丢弃 1 条 rl_steps=21) |
| Train 最终大小 | 4864 (76 batches) |
| Val 最终大小 | 768 (6 batches) |
| 分布匹配误差 | Train < 0.03%, Val < 0.15% |
| 验证结果 | PASSED |