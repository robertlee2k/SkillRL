# 合并训练Checkpoint到推演模型

训练产生的FSDP分片checkpoint需要合并为HuggingFace格式才能在sandbox推演中加载。

## 1. 查找最新Checkpoint

训练checkpoint存放在：
```
/home/bo.li/data/SkillRL/checkpoints/customer_service/grpo_baseline/
```

目录结构：
```
grpo_baseline/
├── global_step_30/
│   └── actor/           # actor模型
├── global_step_60/
│   └── actor/
├── global_step_90/
│   └── actor/
├── global_step_120/
│   └── actor/
├── global_step_150/
│   └── actor/
└── latest_checkpointed_iteration.txt   # 记录最新step编号
```

查看最新checkpoint：
```bash
# 查看最新迭代编号
cat /home/bo.li/data/SkillRL/checkpoints/customer_service/grpo_baseline/latest_checkpointed_iteration.txt

# 或直接列出所有global_step目录
ls -la /home/bo.li/data/SkillRL/checkpoints/customer_service/grpo_baseline/
```

actor目录内是FSDP分片文件（8个rank）：
```
actor/
├── model_world_size_8_rank_0.pt   # 模型权重分片
├── model_world_size_8_rank_1.pt
├── ...
├── model_world_size_8_rank_7.pt
├── optim_world_size_8_rank_*.pt   # 优化器状态（合并时不需要）
├── config.json                    # 模型配置
├── tokenizer.json                 # tokenizer文件
└── ...
```

## 2. 合并Checkpoint

使用 `scripts/model_merger.py` 合并FSDP分片：

```bash
python scripts/model_merger.py merge \
  --backend fsdp \
  --local_dir /home/bo.li/data/SkillRL/checkpoints/customer_service/grpo_baseline/global_step_{STEP}/actor \
  --target_dir /home/bo.li/data/SkillRL/skillrl_models/customer_service/step_{STEP}
```

示例（合并global_step_150）：
```bash
python scripts/model_merger.py merge \
  --backend fsdp \
  --local_dir /home/bo.li/data/SkillRL/checkpoints/customer_service/grpo_baseline/global_step_150/actor \
  --target_dir /home/bo.li/data/SkillRL/skillrl_models/customer_service/step_150
```

输出日志：
```
Got device mesh tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32), mesh_dim_names ('fsdp',)
Processing model shards with 8 (8,) in total
Loading 8 FSDP shards: 100%|██████████| 8/8 [00:04<00:00]
Saving model to /home/bo.li/data/SkillRL/skillrl_models/customer_service/step_150
Saving tokenizer to /home/bo.li/data/SkillRL/skillrl_models/customer_service/step_150
```

## 3. 输出位置

合并后的HF格式模型存放在：
```
/home/bo.li/data/SkillRL/skillrl_models/customer_service/
```

目录结构：
```
customer_service/
├── epoch_160/         # 旧版本模型
├── step_60/           # step_60合并模型
├── step_150/          # step_150合并模型（最新）
│   ├── model-00001-of-00004.safetensors
│   ├── model-00002-of-00004.safetensors
│   ├── model-00003-of-00004.safetensors
│   ├── model-00004-of-00004.safetensors
│   ├── model.safetensors.index.json
│   ├── config.json
│   ├── tokenizer.json
│   └── ...
└── hf_checkpoints/    # 基座模型
```

## 4. 更新Sandbox配置

修改 `viewer/config.py` 中的默认模型路径：

```python
# 沙盒配置
DEFAULT_CHECKPOINT_PATH = "/home/bo.li/data/SkillRL/skillrl_models/customer_service/step_150"
```

重启viewer服务后生效：
```bash
# 如果服务在运行，先杀掉旧进程
lsof -ti:8000 | xargs -r kill -9

# 启动服务
uvicorn viewer.main:app --reload
```

## 5. 一键脚本（可选）

可以创建一个快捷脚本 `scripts/merge_latest_checkpoint.sh`：

```bash
#!/bin/bash
# 合并最新checkpoint并更新配置

CHECKPOINT_DIR="/home/bo.li/data/SkillRL/checkpoints/customer_service/grpo_baseline"
MODEL_DIR="/home/bo.li/data/SkillRL/skillrl_models/customer_service"

# 获取最新step
LATEST=$(cat $CHECKPOINT_DIR/latest_checkpointed_iteration.txt)
echo "Latest checkpoint: global_step_$LATEST"

# 合并
python scripts/model_merger.py merge \
  --backend fsdp \
  --local_dir $CHECKPOINT_DIR/global_step_$LATEST/actor \
  --target_dir $MODEL_DIR/step_$LATEST

# 更新配置
sed -i "s|DEFAULT_CHECKPOINT_PATH = .*|DEFAULT_CHECKPOINT_PATH = \"$MODEL_DIR/step_$LATEST\"|" viewer/config.py

echo "Done! Model saved to $MODEL_DIR/step_$LATEST"
echo "Please restart viewer service to use new model."
```