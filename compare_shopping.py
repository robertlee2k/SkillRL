import os
import re
import ray
from omegaconf import OmegaConf
from functools import partial
from vllm import LLM, SamplingParams

# --- 导入原汁原味的 RL 训练底层组件 ---
from agent_system.environments.env_package.webshop.envs import build_webshop_envs
from agent_system.environments.env_package.webshop import webshop_projection
from agent_system.environments.env_manager import WebshopEnvironmentManager


def extract_action(text: str) -> str:
    match = re.search(r'<action>(.*?)</action>', text, re.IGNORECASE | re.DOTALL)
    if match: return match.group(1).strip()
    fallback = re.search(r'(search\[.*?\]|click\[.*?\])', text, re.IGNORECASE)
    return fallback.group(1).strip() if fallback else "click[back to search]"


def main():
    print("🚀 [1/4] 正在初始化 Ray 引擎...")
    if not ray.is_initialized(): ray.init(ignore_reinit_error=True)

    # ==========================================
    # 核心配置区
    # ==========================================
    PATH_3B_MODEL = "/home/bo.li/data/SkillRL/checkpoints/verl_agent_webshop/3b_hf_merged"
    PATH_3B_SKILL = "/home/bo.li/data/SkillRL/checkpoints/verl_agent_webshop/grpo_qwen2.5_3b_skills_dynamic/updated_skills_step90.json"

    PATH_7B_MODEL = "/home/bo.li/data/SkillRL/checkpoints/verl_agent_webshop/7b_hf_merged"
    PATH_7B_SKILL = "/home/bo.li/data/SkillRL/checkpoints/verl_agent_webshop/grpo_qwen2.5_7b_skills_dynamic/updated_skills_step153.json"

    print("🚀 [2/4] 正在加载 3B 和 7B 双模型 (各限制 40% 显存)...")
    llm_3b = LLM(model=PATH_3B_MODEL, tensor_parallel_size=1, gpu_memory_utilization=0.4, trust_remote_code=True)
    llm_7b = LLM(model=PATH_7B_MODEL, tensor_parallel_size=1, gpu_memory_utilization=0.4, trust_remote_code=True)

    # 【铁证修复】：严格采用训练真实 Eval 温度 0.4！
    sampling_params = SamplingParams(temperature=0.4, top_p=0.9, max_tokens=768, stop=["<|im_end|>"])

    print("🚀 [3/4] 正在加载你最新的真实训练 Config...")
    # 直接加载你 3月16日 最新的那次真实训练配置！
    REAL_CONFIG_PATH = "/home/bo.li/SkillRL/outputs/2026-03-16/22-09-25/.hydra/config.yaml"
    base_config = OmegaConf.load(REAL_CONFIG_PATH)

    # 复制配置，仅覆盖技能库路径，确保 100% 还原训练环境
    config_3b = base_config.copy()
    config_3b.env.use_skills_only_memory = True
    config_3b.env.skills_only_memory.skills_json_path = PATH_3B_SKILL
    config_3b.env.resources_per_worker = {'num_cpus': 1}  # 覆盖资源分配，防止单机环境起不来

    config_7b = base_config.copy()
    config_7b.env.use_skills_only_memory = True
    config_7b.env.skills_only_memory.skills_json_path = PATH_7B_SKILL
    config_7b.env.resources_per_worker = {'num_cpus': 1}

    # 生成基础环境
    raw_env_3b = build_webshop_envs(seed=42, env_num=1, group_n=1, resources_per_worker={'num_cpus': 1}, is_train=False)
    raw_env_7b = build_webshop_envs(seed=42, env_num=1, group_n=1, resources_per_worker={'num_cpus': 1}, is_train=False)

    # 接入原生的 Environment Manager
    proj_func = partial(webshop_projection)
    mgr_3b = WebshopEnvironmentManager(raw_env_3b, proj_func, config_3b)
    mgr_7b = WebshopEnvironmentManager(raw_env_7b, proj_func, config_7b)

    print("🚀 [4/4] 正在调用 Manager 获取原生 Prompt...")
    obs_dict_3b, info_3b = mgr_3b.reset(kwargs={})
    obs_dict_7b, info_7b = mgr_7b.reset(kwargs={})

    task_desc = mgr_3b.tasks[0]

    print("=" * 80)
    print(f"🎯 原生竞技场目标任务: \033[92m{task_desc}\033[0m")
    print("=" * 80)

    # 状态追踪器
    # 这里的 obs 直接就是底层已经完美拼装好 WEBSHOP_TEMPLATE_WITH_MEMORY 的文本
    state = {
        '3B': {'prompt_body': obs_dict_3b['text'][0], 'done': False, 'reward': 0.0, 'step': 1, 'color': '\033[96m'},
        '7B': {'prompt_body': obs_dict_7b['text'][0], 'done': False, 'reward': 0.0, 'step': 1, 'color': '\033[95m'}
    }

    while not (state['3B']['done'] and state['7B']['done']):
        for model_name, llm, mgr in [('3B', llm_3b, mgr_3b), ('7B', llm_7b, mgr_7b)]:
            s = state[model_name]
            if s['done']: continue

            c = s['color']

            if s['step'] == 1 and model_name == '3B':
                print("\n" + "=" * 40 + " [上帝视角] 原生环境生成的完整 Prompt " + "=" * 40)
                print(s['prompt_body'])
                print("=" * 100 + "\n")

            # 严格遵照 Qwen2.5 的 Chat Template
            qwen_prompt = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{s['prompt_body']}<|im_end|>\n<|im_start|>assistant\n"

            print(f"\n{c}[{model_name} - Step {s['step']}]\033[0m 🧠 正在思考...")
            outputs = llm.generate([qwen_prompt], sampling_params, use_tqdm=False)
            response_text = outputs[0].outputs[0].text

            think_match = re.search(r'<think>(.*?)</think>', response_text, re.IGNORECASE | re.DOTALL)
            if think_match:
                print(f"\033[90m{think_match.group(1).strip()}\033[0m")

            action = extract_action(response_text)
            print(f"{c}👉 {model_name} 决定执行: {action}\033[0m")

            # 将文本动作直接丢给原生的 Manager
            next_obs_dict, reward_list, done_list, info_list = mgr.step([action])

            s['prompt_body'] = next_obs_dict['text'][0]
            s['reward'] = reward_list[0]
            s['done'] = done_list[0]

            if s['done']:
                print(f"{c}======================================\033[0m")
                print(f"{c}🏆 {model_name} 购物结束！原生环境裁定最终得分: {s['reward']} / 10.0\033[0m")
                print(f"{c}======================================\033[0m")

            s['step'] += 1
            if s['step'] > 15 and not s['done']:
                print(f"{c}❌ {model_name} 陷入死循环，强制终止！\033[0m")
                s['done'] = True


if __name__ == "__main__":
    main()