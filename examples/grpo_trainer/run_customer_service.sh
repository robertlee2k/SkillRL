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
# 3. 实验规模设置
# ==========================================
num_cpus_per_env_worker=0.1  # The CPU resource allocated for each environment worker.

# 预处理客服场景数据 （恢复长session后共5784条）
# 注意：val_data_size 必须能被 val_batch_size 整除！
# val_batch_size=128, 所以 val_data_size 应该是 128 的倍数 (例如 128, 256, 384, ..., 1536)
python3 scripts/prepare_cs_data.py \
    --playbook_path outputs/playbooks_all.json \
    --output_dir $HOME/data/verl-agent/customer_service \
    --train_data_size 4248 \
    --val_data_size 1536

# 【合理的 Batch Size，加速单步迭代】
#  │ 数据集 │ 数量 │    presale    │  aftersale  │  unknown   │ logistics  │    有订单    │
#  ├───────┼──────┼──────────────┼─────────────┼────────────┼────────────┼──────────────┤
#  │ 训练集 │ 4000 │ 3150 (78.8%) │ 568 (14.2%) │ 179 (4.5%) │ 103 (2.6%) │ 1071 (26.8%) │
#  ├───────┼──────┼──────────────┼─────────────┼────────────┼────────────┼──────────────┤
#  │ 验证集 │ 1784 │ 1434 (80.4%) │ 229 (12.8%) │ 77 (4.3%)  │ 44 (2.5%)  │ 486 (27.2%)
train_data_size=64  # 我们的train数据有4000条，估计2500条就能收敛
val_data_size=128
group_size=8         # Parallel rollouts per episode

# ==========================================
# 4. 启动 verl GRPO 训练
# 【修改2：增加生成长度max_response_length到1024，防止思考被截断】
# 【修改3：关闭前置兜底，激活环境内的耐心系统】
# 【修改4: 压住KL】
# 【修改5: 改成bf16】
# 【修改6: prompt长度翻倍到8192， max_num_batched_tokens到16384】
# ==========================================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/customer_service/train.parquet \
    data.val_files=$HOME/data/verl-agent/customer_service/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=16384 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.1 \
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
    env.seed=0 \
    env.max_steps=20 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +env.use_fallback_projection=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_agent_customer_service' \
    trainer.experiment_name='grpo_baseline' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    +trainer.val_freq=10 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.default_local_dir=/home/bo.li/data/SkillRL/checkpoints/customer_service/grpo_baseline "$@"