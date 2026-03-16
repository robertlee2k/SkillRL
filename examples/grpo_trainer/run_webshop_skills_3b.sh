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
# 2. 模型、数据与 WandB 配置 (请填入你的信息)
# ==========================================
# 指向你在本地下载好的 Qwen2.5-7B 模型目录
export MODEL_PATH="/home/bo.li/data/models/Qwen2.5-3B-Instruct"
# 指向你在本地下载好的 Embedding 模型目录
export EMBEDDING_PATH="/home/bo.li/data/models/bge-small-en-v1.5"

# 实验记录器配置
export WANDB_API_KEY="7dfb306b2aa3742f6839"  # <--- 请务必填入你的 WandB Key
export WANDB_NAME="webshop_grpo_qwen2.5_7b_sft_skills_dynamic"

# ==========================================
# 3. 实验规模设置
# ==========================================
num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker.

train_data_size=16  # Moderate size
val_data_size=64    # Moderate size
group_size=8        # Moderate parallelism

# 预处理数据 (确保与实际项目里的 parquet 文件路径一致)
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# ==========================================
# 4. 启动 verl GRPO 训练 (针对 8张 A100 魔改)
# ==========================================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=6000 \
    data.max_response_length=768 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=/home/bo.li/SkillRL/memory_data/webshop/claude_style_skills.json \
    +env.skills_only_memory.retrieval_mode=embedding \
    +env.skills_only_memory.embedding_model_path=$EMBEDDING_PATH \
    +env.skills_only_memory.top_k=6 \
    +env.skills_only_memory.enable_dynamic_update=True \
    +env.skills_only_memory.update_threshold=0.4 \
    +env.skills_only_memory.max_new_skills=3 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name='grpo_qwen2.5_3b_skills_dynamic' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    +trainer.val_freq=10 \
    trainer.total_epochs=50 \
    trainer.val_before_train=True \
    trainer.default_local_dir=/home/bo.li/data/SkillRL/checkpoints/verl_agent_webshop/grpo_qwen2.5_3b_skills_dynamic "$@"