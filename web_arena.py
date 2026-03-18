import os
import re
import ray
import json
import asyncio
from omegaconf import OmegaConf
from functools import partial
from vllm import LLM, SamplingParams

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from agent_system.environments.env_package.webshop.envs import build_webshop_envs
from agent_system.environments.env_package.webshop import webshop_projection
from agent_system.environments.env_manager import WebshopEnvironmentManager

app = FastAPI()

# 全局环境缓存，防止批量测试时 Ray Actor 重复创建导致显存溢出
sys_env = {}


def extract_action(text: str) -> str:
    match = re.search(r'<action>(.*?)</action>', text, re.IGNORECASE | re.DOTALL)
    if match: return match.group(1).strip()
    fallback = re.search(r'(search\[.*?\]|click\[.*?\])', text, re.IGNORECASE)
    return fallback.group(1).strip() if fallback else "click[back to search]"


def patch_step1_prompt(mgr, raw_prompt):
    if mgr.retrieval_memory and mgr.retrieved_memories:
        skills_text = mgr.retrieval_memory.format_for_prompt(mgr.retrieved_memories[0])
        patch = f"\n\n## Retrieved Relevant Experience\n\n{skills_text}\n\n"
        if "Your current observation is:" in raw_prompt:
            parts = raw_prompt.split("Your current observation is:")
            return parts[0] + patch + "Your current observation is:" + parts[1]
    return raw_prompt


def parse_screen_elements(raw_obs):
    clean_obs = str(raw_obs).replace("'", "")
    elements = [e.strip() for e in clean_obs.split('[SEP]') if e.strip()]
    return elements


def parse_skills(prompt_body):
    match = re.search(r'## Retrieved Relevant Experience(.*?)(?=## Current Progress|Your current observation is:)',
                      prompt_body, re.IGNORECASE | re.DOTALL)
    if match:
        skills_text = match.group(1).strip()
        return [line.strip() for line in skills_text.split('\n') if line.strip()]
    return []


@app.on_event("startup")
def startup_event():
    print("🚀 正在全局初始化 Web 竞技场环境与模型 (仅执行一次)...")
    if not ray.is_initialized():
        # 仅将海量的 Object Spilling 数据放到大容量磁盘
        SPILL_DIR = "/polarfs/models/luotuo/ray_tmp"  # 👈 请依然替换为你的真实大盘路径
        os.makedirs(SPILL_DIR, exist_ok=True)

        ray.init(
            ignore_reinit_error=True,
            # ❌ 删掉之前加的 _temp_dir 参数，让核心 socket 继续留在本地 /tmp
            _system_config={
                # ✅ 强制把内存放不下的几个 GB/TB 的大文件写到大盘里
                "object_spilling_config": json.dumps({
                    "type": "filesystem",
                    "params": {"directory_path": SPILL_DIR}
                })
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

    # 全局持有一份原生环境，避免批量评测时内存泄漏
    raw_env_3b = build_webshop_envs(seed=42, env_num=1, group_n=1, resources_per_worker={'num_cpus': 1}, is_train=False)
    raw_env_7b = build_webshop_envs(seed=42, env_num=1, group_n=1, resources_per_worker={'num_cpus': 1}, is_train=False)
    proj_func = partial(webshop_projection)

    sys_env['mgr_3b'] = WebshopEnvironmentManager(raw_env_3b, proj_func, config_3b)
    sys_env['mgr_7b'] = WebshopEnvironmentManager(raw_env_7b, proj_func, config_7b)
    print("✅ 全局初始化完成！后端已就绪。")


@app.get("/")
def get_dashboard():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


def reset_environments():
    obs_3b, _ = sys_env['mgr_3b'].reset(kwargs={})
    obs_7b, _ = sys_env['mgr_7b'].reset(kwargs={})
    return obs_3b, obs_7b


def step_environment(mgr, response_text):
    return mgr.step([response_text])


@app.websocket("/ws/arena")
async def websocket_arena(websocket: WebSocket):
    await websocket.accept()
    try:
        # 接收前端的参数：跑多少个批量任务
        msg = await websocket.receive_json()
        num_tasks = int(msg.get("num_tasks", 10))

        score_3b = 0
        score_7b = 0

        # ========== 开启批量评测循环 ==========
        for task_idx in range(num_tasks):
            # 刷新环境获取新任务（测试集会自动翻页）
            obs_dict_3b, obs_dict_7b = await asyncio.to_thread(reset_environments)
            task_desc = sys_env['mgr_3b'].tasks[0]

            # 通知前端新一轮任务开始
            await websocket.send_json({
                "type": "task_start",
                "task": task_desc,
                "current": task_idx + 1,
                "total": num_tasks
            })

            state = {
                '3B': {'prompt_body': patch_step1_prompt(sys_env['mgr_3b'], obs_dict_3b['text'][0]),
                       'raw_obs': obs_dict_3b['anchor'][0], 'done': False, 'step': 1},
                '7B': {'prompt_body': patch_step1_prompt(sys_env['mgr_7b'], obs_dict_7b['text'][0]),
                       'raw_obs': obs_dict_7b['anchor'][0], 'done': False, 'step': 1}
            }

            # 单个任务的回合交互
            while not (state['3B']['done'] and state['7B']['done']):
                for model_name, llm, mgr in [('3B', sys_env['llm_3b'], sys_env['mgr_3b']),
                                             ('7B', sys_env['llm_7b'], sys_env['mgr_7b'])]:
                    s = state[model_name]
                    if s['done']: continue

                    # 1. 渲染屏幕
                    await websocket.send_json({
                        "type": "state_update",
                        "model": model_name,
                        "step": s['step'],
                        "screen_elements": parse_screen_elements(s['raw_obs']),
                        "skills": parse_skills(s['prompt_body'])
                    })

                    # 2. 推理
                    qwen_prompt = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{s['prompt_body']}<|im_end|>\n<|im_start|>assistant\n"
                    loop = asyncio.get_event_loop()
                    outputs = await loop.run_in_executor(None,
                                                         lambda: llm.generate([qwen_prompt], sys_env['sampling_params'],
                                                                              use_tqdm=False))
                    response_text = outputs[0].outputs[0].text

                    think_match = re.search(r'<think>(.*?)</think>', response_text, re.IGNORECASE | re.DOTALL)
                    think_text = think_match.group(1).strip() if think_match else "No think tags."
                    action_text = extract_action(response_text)

                    # 3. 渲染动作
                    await websocket.send_json({
                        "type": "model_action",
                        "model": model_name,
                        "step": s['step'],
                        "think": think_text,
                        "action": action_text
                    })

                    # 4. 执行原汁原味的步进
                    next_obs_dict, reward_list, done_list, _ = await asyncio.to_thread(step_environment, mgr,
                                                                                       response_text)

                    s['prompt_body'] = next_obs_dict['text'][0]
                    s['raw_obs'] = next_obs_dict['anchor'][0]
                    s['done'] = done_list[0]

                    if s['done']:
                        reward = float(reward_list[0])
                        if reward == 10.0:
                            if model_name == '3B': score_3b += 1
                            if model_name == '7B': score_7b += 1
                        await websocket.send_json({"type": "task_done", "model": model_name, "reward": reward})

                    s['step'] += 1
                    if s['step'] > 15 and not s['done']:
                        s['done'] = True
                        await websocket.send_json(
                            {"type": "task_done", "model": model_name, "reward": 0.0, "msg": "Timeout"})

            # 单个任务结束后，刷新全局计分板
            await websocket.send_json({
                "type": "batch_progress",
                "score_3b": score_3b,
                "score_7b": score_7b,
                "current": task_idx + 1
            })

    except WebSocketDisconnect:
        print("⚠️ 前端断开连接，批量任务终止。")
    except Exception as e:
        print(f"❌ 发生异常: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=28008)