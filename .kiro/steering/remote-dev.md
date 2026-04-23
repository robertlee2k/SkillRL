---
inclusion: auto
---

# Remote Development: Mac + A100 Server

This project uses a **local Mac editing + remote A100 execution** workflow. All code editing happens locally in Kiro; all GPU execution happens on the remote server via SSH.

## Connection Info

| Item | Value |
|------|-------|
| SSH alias | `a100` (configured in `~/.ssh/config`) |
| Conda env | `skillrl` |
| Remote repo | `~/SkillRL/` |

## Workflow: Edit → Sync → Run

### 1. Edit locally

Make all code changes in Kiro on the Mac. Never edit files directly on the server.

### 2. Sync to remote

```bash
./sync_to_a100.sh
```

This runs `rsync` from local to remote, excluding `.git/`, `__pycache__/`, `.DS_Store`, `.idea/`, `.superpowers/`, `wandb/`, and `*.log`. The `--delete` flag ensures removed files are also removed on the server.

For a dry run (preview only):
```bash
./sync_to_a100.sh --dry-run
```

### 3. Run on remote

All remote commands must activate conda first. Use this pattern:

```bash
ssh a100 "source ~/miniconda3/etc/profile.d/conda.sh && conda activate skillrl && cd ~/SkillRL && <your command>"
```

For GPU-specific tasks, prefix with `CUDA_VISIBLE_DEVICES`:

```bash
ssh a100 "source ~/miniconda3/etc/profile.d/conda.sh && conda activate skillrl && cd ~/SkillRL && CUDA_VISIBLE_DEVICES=0,1,2,3 python your_script.py"
```

### 4. Pull results back (when needed)

```bash
rsync -avz a100:~/SkillRL/outputs/ ./outputs/ --exclude '.git/'
```

## GPU Layout

The server has 8× A100-SXM4-80GB (GPUs 0–7). Check current usage before running anything heavy:

```bash
ssh a100 "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader"
```

Other training jobs may be running on GPUs 4–7. Always confirm which GPUs are free before launching.

## vLLM Notes

vLLM 0.11 with tensor parallelism requires the spawn multiproc method when using `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_WORKER_MULTIPROC_METHOD=spawn python your_script.py --tensor_parallel_size 4
```

Without `VLLM_WORKER_MULTIPROC_METHOD=spawn`, you will get `Cannot re-initialize CUDA in forked subprocess` errors.

## Key Rules

1. **One-way sync**: Local → Remote only. Do not edit code on the server.
2. **Git is independent**: Both sides have `.git/` but rsync excludes it. Use git locally for version control.
3. **Always activate conda**: The system Python on the server does not have the project dependencies. Every SSH command must include `source ~/miniconda3/etc/profile.d/conda.sh && conda activate skillrl`.
4. **Check GPU availability**: Before launching GPU tasks, verify which GPUs are free to avoid conflicts with running training jobs.

## Trained Models & Checkpoints

| Path | Description |
|------|-------------|
| `~/data/models/Qwen2.5-7B-Instruct` | Base model (SFT checkpoint) |
| `~/data/SkillRL/checkpoints/customer_service/grpo_baseline/` | FSDP training checkpoints (global_step_N/) |
| `~/data/SkillRL/skillrl_models/customer_service/` | Merged HF-format models (step_60, step_150, step_210) |
| `~/SkillRL/outputs/playbooks_all_fixed_v2.json` | Playbook data (5732 playbooks, ghost actions fixed) |

## Quick Reference Commands

Check server is reachable:
```bash
ssh a100 "echo OK && hostname"
```

Sync and run a script:
```bash
./sync_to_a100.sh && ssh a100 "source ~/miniconda3/etc/profile.d/conda.sh && conda activate skillrl && cd ~/SkillRL && python scripts/your_script.py"
```

Check training checkpoint status:
```bash
ssh a100 "cat ~/data/SkillRL/checkpoints/customer_service/grpo_baseline/latest_checkpointed_iteration.txt"
```

Monitor GPU usage:
```bash
ssh a100 "nvidia-smi"
```
