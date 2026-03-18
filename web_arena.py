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


def parse_skills_from_think(think_text):
    # 正则提取类似 [Skill_01] 或 [Skill_Search] 的标签
    return list(set(re.findall(r'\[Skill_[\w\d]+\]', think_text)))

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
        msg = await websocket.receive_json()
        num_tasks = int(msg.get("num_tasks", 10))

        # 统计数据结构
        stats = {"3B": {"success": 0, "skill_usage": {}}, "7B": {"success": 0, "skill_usage": {}}}

        for task_idx in range(num_tasks):
            # 增加空字典作为 kwargs 传入
            obs_3b, obs_7b = await asyncio.to_thread(lambda: (
                sys_env['mgr_3b'].reset(kwargs={})[0],
                sys_env['mgr_7b'].reset(kwargs={})[0]
            ))
            task_desc = sys_env['mgr_3b'].tasks[0]

            # 记录当前 Task 的完整轨迹供回放
            task_history = {
                "id": task_idx + 1,
                "goal": task_desc,
                "steps": {"3B": [], "7B": []},
                "final_reward": {"3B": 0, "7B": 0}
            }

            await websocket.send_json(
                {"type": "task_start", "task": task_desc, "current": task_idx + 1, "total": num_tasks})

            active = {"3B": True, "7B": True}
            states = {
                "3B": {"obs": obs_3b, "step": 1},
                "7B": {"obs": obs_7b, "step": 1}
            }

            while any(active.values()):
                for m in ["3B", "7B"]:
                    if not active[m]: continue

                    mgr = sys_env[f'mgr_{m.lower()}']
                    # 强引导 Prompt：要求模型必须引用 Skill ID
                    prompt = f"System: You are an agent. Reference skills as [Skill_XX] in <think>.\nGoal: {task_desc}\nObs: {states[m]['obs']['text'][0]}"

                    outputs = await asyncio.get_event_loop().run_in_executor(None, lambda: sys_env[
                        f'llm_{m.lower()}'].generate([prompt], sys_env['sampling_params'], use_tqdm=False))
                    resp = outputs[0].outputs[0].text

                    # 1. 提取内容（增加防御性判断，防止 NoneType 报错）
                    think_match = re.search(r'<think>(.*?)</think>', resp, re.S)
                    think = think_match.group(1).strip() if think_match else resp.split('<action>')[0].strip()

                    # 2. 提取 Action（同样增加防御）
                    action_match = re.search(r'<action>(.*?)</action>', resp, re.S)
                    action = action_match.group(1).strip() if action_match else "click[back to search]"

                    # 3. 解析 Skills（扩大搜索范围到全文）
                    skills_detected = parse_skills_from_think(resp)

                    # 4. 发送实时更新 (Live Update)
                    await websocket.send_json({
                        "type": "live_step_update",
                        "model": m,
                        "data": {
                            "step": states[m]['step'],
                            "screen": states[m]['obs']['text'][0].replace('[SEP]', '\n'),  # 换行处理
                            "think": think,
                            "action": action,
                            "skills": skills_detected
                        }
                    })

                    # 执行步进...
                    next_obs, reward, done, _ = await asyncio.to_thread(lambda: mgr.step([resp]))
                    states[m]["obs"] = next_obs
                    states[m]["step"] += 1

                    if done[0] or states[m]["step"] > 15:
                        active[m] = False
                        task_history["final_reward"][m] = reward[0]
                        if reward[0] == 10.0: stats[m]["success"] += 1
                        # 更新全局技能热力统计
                        for s in skills_detected:
                            stats[m]["skill_usage"][s] = stats[m]["skill_usage"].get(s, 0) + (
                                1 if reward[0] == 10.0 else 0)

            # 任务结束，发送完整回放包
            await websocket.send_json({
                "type": "history_entry",
                "data": task_history,
                "stats": stats
            })

    except WebSocketDisconnect:
        print("⚠️ 前端断开连接，批量任务终止。")
    except Exception as e:
        print(f"❌ 发生异常: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=28008)