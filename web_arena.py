import os
import re
import ray
import json
import asyncio
import numpy as np
from omegaconf import OmegaConf
from functools import partial
from vllm import LLM, SamplingParams
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from agent_system.environments.env_package.webshop.envs import build_webshop_envs
from agent_system.environments.env_package.webshop import webshop_projection
from agent_system.environments.env_manager import WebshopEnvironmentManager

sys_env = {}


def to_json_serializable(data):
    """递归将所有 numpy 数据类型转换为 python 原生类型，解决 JSON 报错"""
    if isinstance(data, dict):
        return {k: to_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_json_serializable(i) for i in data]
    elif isinstance(data, (np.integer, np.int64)):
        return int(data)
    elif isinstance(data, (np.floating, np.float64)):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data


def extract_action(text: str) -> str:
    match = re.search(r'<action>(.*?)</action>', text, re.IGNORECASE | re.DOTALL)
    if match: return match.group(1).strip()
    fallback = re.search(r'(search\[.*?\]|click\[.*?\])', text, re.IGNORECASE)
    return fallback.group(1).strip() if fallback else "click[back to search]"


def patch_step1_prompt(mgr, raw_prompt):
    """你的原始补丁逻辑：首步注入 RAG"""
    if mgr.retrieval_memory and mgr.retrieved_memories:
        skills_text = mgr.retrieval_memory.format_for_prompt(mgr.retrieved_memories[0])
        patch = f"\n\n## Retrieved Relevant Experience\n\n{skills_text}\n\n"
        if "Your current observation is:" in raw_prompt:
            parts = raw_prompt.split("Your current observation is:")
            return parts[0] + patch + "Your current observation is:" + parts[1]
    return raw_prompt


def extract_skills_from_prompt(prompt_text):
    """从你的 Prompt 结构中提取技能名称（匹配 **加粗文本**）"""
    skills = []
    exp_match = re.search(r'## Retrieved Relevant Experience(.*?)## Current Progress', prompt_text, re.S)
    if exp_match:
        skills = re.findall(r'- \*\*(.*?)\*\*:', exp_match.group(1))
    return list(set(skills))


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 正在全局初始化 Web 竞技场环境与模型 (仅执行一次)...")
    if not ray.is_initialized():
        SPILL_DIR = "/polarfs/models/luotuo/ray_tmp"
        os.makedirs(SPILL_DIR, exist_ok=True)
        ray.init(
            ignore_reinit_error=True,
            _system_config={
                "object_spilling_config": json.dumps({"type": "filesystem", "params": {"directory_path": SPILL_DIR}})
            }
        )

    PATH_3B_MODEL = "/home/bo.li/data/SkillRL/checkpoints/verl_agent_webshop/3b_hf_merged"
    PATH_3B_SKILL = "/home/bo.li/data/SkillRL/checkpoints/verl_agent_webshop/grpo_qwen2.5_3b_skills_dynamic/updated_skills_step90.json"
    PATH_7B_MODEL = "/home/bo.li/data/SkillRL/checkpoints/verl_agent_webshop/7b_hf_merged"
    PATH_7B_SKILL = "/home/bo.li/data/SkillRL/checkpoints/verl_agent_webshop/grpo_qwen2.5_7b_skills_dynamic/updated_skills_step153.json"

    sys_env['llm_3b'] = LLM(model=PATH_3B_MODEL, tensor_parallel_size=1, gpu_memory_utilization=0.4,
                            trust_remote_code=True)
    sys_env['llm_7b'] = LLM(model=PATH_7B_MODEL, tensor_parallel_size=1, gpu_memory_utilization=0.4,
                            trust_remote_code=True)
    sys_env['sampling_params'] = SamplingParams(temperature=0.4, top_p=0.9, max_tokens=768, stop=["<|im_end|>"])

    REAL_CONFIG_PATH = "/home/bo.li/SkillRL/outputs/2026-03-16/22-09-25/.hydra/config.yaml"
    base_config = OmegaConf.load(REAL_CONFIG_PATH)

    config_3b, config_7b = base_config.copy(), base_config.copy()
    config_3b.env.update({'use_skills_only_memory': True, 'resources_per_worker': {'num_cpus': 1}})
    config_3b.env.skills_only_memory.skills_json_path = PATH_3B_SKILL
    config_7b.env.update({'use_skills_only_memory': True, 'resources_per_worker': {'num_cpus': 1}})
    config_7b.env.skills_only_memory.skills_json_path = PATH_7B_SKILL

    sys_env['mgr_3b'] = WebshopEnvironmentManager(
        build_webshop_envs(seed=42, env_num=1, group_n=1, resources_per_worker={'num_cpus': 1}, is_train=False),
        partial(webshop_projection), config_3b)
    sys_env['mgr_7b'] = WebshopEnvironmentManager(
        build_webshop_envs(seed=42, env_num=1, group_n=1, resources_per_worker={'num_cpus': 1}, is_train=False),
        partial(webshop_projection), config_7b)
    print("✅ 全局初始化完成！后端已就绪。")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def get_dashboard():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws/arena")
async def websocket_arena(websocket: WebSocket):
    await websocket.accept()
    try:
        msg = await websocket.receive_json()
        num_tasks = int(msg.get("num_tasks", 10))
        stats = {"3B": {"success": 0, "skill_usage": {}}, "7B": {"success": 0, "skill_usage": {}}}

        for task_idx in range(num_tasks):
            obs_3b, obs_7b = await asyncio.to_thread(
                lambda: (sys_env['mgr_3b'].reset(kwargs={})[0], sys_env['mgr_7b'].reset(kwargs={})[0]))
            task_desc = sys_env['mgr_3b'].tasks[0]

            task_history = {"id": task_idx + 1, "goal": task_desc, "steps": {"3B": [], "7B": []},
                            "final_reward": {"3B": 0, "7B": 0}}
            await websocket.send_json(
                {"type": "task_start", "task": task_desc, "current": task_idx + 1, "total": num_tasks})

            active = {"3B": True, "7B": True}

            # 使用 prompt_body 记录状态，使用 anchor 记录屏幕
            states = {
                "3B": {"prompt": patch_step1_prompt(sys_env['mgr_3b'], obs_3b['text'][0]),
                       "anchor": obs_3b['anchor'][0], "step": 1},
                "7B": {"prompt": patch_step1_prompt(sys_env['mgr_7b'], obs_7b['text'][0]),
                       "anchor": obs_7b['anchor'][0], "step": 1}
            }

            while any(active.values()):
                for m in ["3B", "7B"]:
                    if not active[m]: continue

                    mgr = sys_env[f'mgr_{m.lower()}']
                    llm = sys_env[f'llm_{m.lower()}']

                    # 使用 Qwen 的严格 Chat Template
                    prompt_body = states[m]['prompt']
                    qwen_prompt = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt_body}<|im_end|>\n<|im_start|>assistant\n"

                    outputs = await asyncio.get_event_loop().run_in_executor(None, lambda: llm.generate([qwen_prompt],
                                                                                                        sys_env[
                                                                                                            'sampling_params'],
                                                                                                        use_tqdm=False))
                    resp = outputs[0].outputs[0].text if outputs else ""

                    think_match = re.search(r'<think>(.*?)</think>', resp, re.S)
                    think = think_match.group(1).strip() if think_match else resp.split('<action>')[0].strip()
                    action = extract_action(resp)

                    # 从系统 Prompt 提取出检索到的 Skills
                    retrieved_skills = extract_skills_from_prompt(prompt_body)
                    # 检查大模型在 Think 中是否真的用到了这个 Skill 的名字
                    used_skills = [s for s in retrieved_skills if s.lower() in think.lower()]

                    step_data = {
                        "step": int(states[m]['step']),
                        "screen": str(states[m]['anchor']).replace('[SEP]', '\n').replace("'", ""),
                        # 👈 恢复 anchor 作为虚拟屏幕
                        "think": think,
                        "action": action,
                        "skills": retrieved_skills,
                        "used_skills": used_skills
                    }
                    task_history["steps"][m].append(step_data)

                    await websocket.send_json(
                        {"type": "live_step_update", "model": m, "data": to_json_serializable(step_data)})

                    next_obs, reward, done, _ = await asyncio.to_thread(lambda: mgr.step([resp]))
                    states[m]["prompt"] = next_obs['text'][0]
                    states[m]["anchor"] = next_obs['anchor'][0]
                    states[m]["step"] += 1

                    if done[0] or states[m]["step"] > 15:
                        active[m] = False
                        r_val = float(reward[0])
                        task_history["final_reward"][m] = r_val
                        if r_val >= 10.0: stats[m]["success"] += 1
                        for s in used_skills:
                            stats[m]["skill_usage"][s] = stats[m]["skill_usage"].get(s, 0) + (1 if r_val >= 10.0 else 0)

            await websocket.send_json({"type": "history_entry", "data": to_json_serializable(task_history),
                                       "stats": to_json_serializable(stats)})

    except Exception as e:
        print(f"❌ 运行异常: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=28008)