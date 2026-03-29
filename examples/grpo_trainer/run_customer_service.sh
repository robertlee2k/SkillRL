#!/bin/bash
set -x

# ==========================================
# 1. 基础引擎与日志设置
# ==========================================
ENGINE=${1:-vllm}
shift  # Remove first argument so $@ only contains extra params
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Enable more verbose logging
export RAY_BACKEND_LOG_LEVEL=debug
export VLLM_LOGGING_LEVEL=DEBUG

export RAY_INIT_KWARGS='{"_system_config": {"object_spilling_config": "{\"type\": \"filesystem\", \"params\": {\"directory_path\": \"/home/bo.li/data/ray_spill\"}}" }}'


# ==========================================
# 2. 模型与 WandB 配置
# ==========================================
# 指向你在本地下载好的 Qwen2.5-7B 模型目录
export MODEL_PATH="/home/bo.li/data/models/Qwen2.5-7B-Instruct"

# 实验记录器配置
export WANDB_API_KEY="7dfb306b2aa3742f6839"  # <--- 请务必填入你的 WandB Key
export WANDB_NAME="customer_service_grpo_baseline"

# ==========================================
# 3. 实验规模设置（统一参数体系）
# ==========================================
num_cpus_per_env_worker=0.1  # The CPU resource allocated for each environment worker.

# 【关键参数定义 - 所有下游命令统一引用】
train_batch_size=64   # 训练批次大小（drop-last对齐）
val_batch_size=128    # 验证批次大小（drop-last对齐）
val_ratio=0.2        # 验证集比例
max_rl_steps=20       # RL steps上限阈值（超过的playbook被丢弃）
seed=1688               # 随机种子
group_size=8          # Parallel rollouts per episode

# 预处理客服场景数据（工业级数据准备引擎）
python3 scripts/prepare_cs_data.py \
    --playbook_path outputs/playbooks_all.json \
    --output_dir $HOME/data/verl-agent/customer_service \
    --val_ratio $val_ratio \
    --train_batch_size $train_batch_size \
    --val_batch_size $val_batch_size \
    --max_rl_steps $max_rl_steps \
    --seed $seed

# ==========================================
# 4. 启动 verl GRPO 训练
# 【修改2：增加生成长度max_response_length到1024，防止思考被截断】
# 【修改3：关闭前置兜底，激活环境内的耐心系统】
# 【修改4: KL惩罚改到0.04】
# 【修改5: 改成bf16】
# 【修改6: prompt长度翻倍到8192， max_num_batched_tokens到16384】
# ==========================================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/customer_service/train.parquet \
    data.val_files=$HOME/data/verl-agent/customer_service/test.parquet \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=16384 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    algorithm.use_kl_in_reward=False \
    env.env_name=CustomerService \
    env.playbook_path=outputs/playbooks_all.json \
    env.seed=0 \
    env.max_steps=$max_rl_steps \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +env.use_fallback_projection=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_agent_customer_service' \
    trainer.experiment_name='grpo_baseline' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=5 \
    +trainer.val_freq=10 \
    trainer.total_epochs=3 \
    trainer.val_before_train=True \
    trainer.default_local_dir=/home/bo.li/data/SkillRL/checkpoints/customer_service/grpo_baseline "$@"