import os, re, ray, json, asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import uvicorn


# 1. 递归转换 Numpy 类型，确保 JSON 序列化万无一失
def np_to_py(obj):
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: np_to_py(v) for k, v in obj.items()}
    if isinstance(obj, list): return [np_to_py(i) for i in obj]
    return obj


# 2. 增强版 Skills 提取
def parse_skills_from_think(text: str):
    if not text: return []
    # 匹配 [Skill_01] 或 Skill_01，忽略大小写
    found = re.findall(r'\[?Skill_[\w\d]+\]?', text, re.I)
    return list(set([s.replace('[', '').replace(']', '') for s in found]))


def extract_action(text: str) -> str:
    match = re.search(r'<action>(.*?)</action>', text, re.S | re.I)
    return match.group(1).strip() if match else "click[back to search]"


# 3. 使用 Lifespan 管理初始化 (取代废弃的 on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 启动时初始化 ---
    print("🚀 系统正在初始化模型与环境...")
    # 这里放入你之前的初始化代码，例如：
    # sys_env['llm_3b'] = ...
    # sys_env['mgr_3b'] = ...
    yield
    # --- 关闭时清理 ---
    print("🛑 系统正在关闭...")


app = FastAPI(lifespan=lifespan)
sys_env = {}  # 假设你的全局变量依然存放在这里


@app.websocket("/ws/arena")
async def websocket_arena(websocket: WebSocket):
    await websocket.accept()
    try:
        msg = await websocket.receive_json()
        num_tasks = int(msg.get("num_tasks", 10))
        stats = {"3B": {"success": 0, "skill_usage": {}}, "7B": {"success": 0, "skill_usage": {}}}

        for task_idx in range(num_tasks):
            # 适配你的 reset 接口
            obs_3b, obs_7b = await asyncio.to_thread(lambda: (
                sys_env['mgr_3b'].reset(kwargs={})[0],
                sys_env['mgr_7b'].reset(kwargs={})[0]
            ))
            task_desc = sys_env['mgr_3b'].tasks[0]

            task_history = {"id": task_idx + 1, "goal": task_desc, "steps": {"3B": [], "7B": []},
                            "final_reward": {"3B": 0, "7B": 0}}
            await websocket.send_json(
                {"type": "task_start", "task": task_desc, "current": task_idx + 1, "total": num_tasks})

            active = {"3B": True, "7B": True}
            states = {"3B": {"obs": obs_3b, "step": 1}, "7B": {"obs": obs_7b, "step": 1}}

            while any(active.values()):
                for m in ["3B", "7B"]:
                    if not active[m]: continue

                    mgr = sys_env[f'mgr_{m.lower()}']
                    llm = sys_env[f'llm_{m.lower()}']

                    # 运行推理
                    prompt = f"Goal: {task_desc}\nObs: {states[m]['obs']['text'][0]}"
                    outputs = await asyncio.get_event_loop().run_in_executor(None, lambda: llm.generate([prompt],
                                                                                                        sys_env[
                                                                                                            'sampling_params'],
                                                                                                        use_tqdm=False))
                    resp = outputs[0].outputs[0].text if outputs else ""

                    # 解析内容
                    think_match = re.search(r'<think>(.*?)</think>', resp, re.S)
                    think = think_match.group(1).strip() if think_match else resp.split('<action>')[0].strip()
                    action = extract_action(resp)
                    skills = parse_skills_from_think(resp)

                    step_info = {
                        "step": int(states[m]['step']),
                        "screen": str(states[m]['obs']['text'][0]).replace('[SEP]', '\n'),
                        "think": think, "action": action, "skills": skills
                    }
                    task_history["steps"][m].append(step_info)

                    # 发送实时步进
                    await websocket.send_json({"type": "live_step_update", "model": m, "data": np_to_py(step_info)})

                    # 执行动作
                    next_obs, reward, done, _ = await asyncio.to_thread(lambda: mgr.step([resp]))
                    states[m]["obs"] = next_obs
                    states[m]["step"] += 1

                    if done[0] or states[m]["step"] > 15:
                        active[m] = False
                        reward_val = float(reward[0])
                        task_history["final_reward"][m] = reward_val
                        if reward_val >= 10.0: stats[m]["success"] += 1
                        for s in skills:
                            stats[m]["skill_usage"][s] = stats[m]["skill_usage"].get(s, 0) + (
                                1 if reward_val >= 10.0 else 0)

            # 任务结束发送历史总包
            await websocket.send_json(
                {"type": "history_entry", "data": np_to_py(task_history), "stats": np_to_py(stats)})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"❌ 运行异常: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=28008)