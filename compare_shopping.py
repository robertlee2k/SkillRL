import os
import re
import ray
from omegaconf import OmegaConf
from functools import partial
from vllm import LLM, SamplingParams

from agent_system.environments.env_package.webshop.envs import build_webshop_envs
from agent_system.environments.env_package.webshop import webshop_projection
from agent_system.environments.env_manager import WebshopEnvironmentManager


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


def render_virtual_screen(model_name, raw_obs, color_code):
    clean_obs = str(raw_obs).replace("'", "")
    elements = [e.strip() for e in clean_obs.split('[SEP]') if e.strip()]

    print(f"\n{color_code}╭{'─' * 20} 📱 {model_name} 眼中的当前网页 {'─' * 20}╮\033[0m")
    for item in elements:
        if "Price:" in item or "price" in item.lower():
            print(f"{color_code}│  💰 \033[1;33m{item}\033[0m")
        elif item.lower() in ['buy now', 'search', 'back to search', '< prev', 'next >']:
            print(f"{color_code}│  🔘 \033[1;37;44m [{item}] \033[0m")
        elif "Instruction:" in item:
            print(f"{color_code}│  📌 \033[1;32m{item}\033[0m")
        else:
            print(f"{color_code}│  📄 {item}\033[0m")
    print(f"{color_code}╰{'─' * 56}╯\033[0m")


def render_skills_panel(prompt_body, color_code):
    """🧠 新增：技能探针，提取并展示当前挂载的 RAG 技能"""
    match = re.search(r'## Retrieved Relevant Experience(.*?)(?=## Current Progress|Your current observation is:)',
                      prompt_body, re.IGNORECASE | re.DOTALL)
    if match:
        skills_text = match.group(1).strip()
        if skills_text:
            print(f"{color_code}╭{'─' * 20} 🎒 当前挂载的动态技能 {'─' * 21}╮\033[0m")
            # 简单排版一下技能条目
            for line in skills_text.split('\n'):
                if line.strip():
                    print(f"{color_code}│ \033[3m{line[:70] + '...' if len(line) > 70 else line}\033[0m")
            print(f"{color_code}╰{'─' * 56}╯\033[0m")


def main():
    print("🚀 [1/4] 正在初始化 Ray 引擎...")
    if not ray.is_initialized(): ray.init(ignore_reinit_error=True)

    # 路径配置
    PATH_3B_MODEL = "/home/bo.li/data/SkillRL/checkpoints/verl_agent_webshop/3b_hf_merged"
    PATH_3B_SKILL = "/home/bo.li/data/SkillRL/checkpoints/verl_agent_webshop/grpo_qwen2.5_3b_skills_dynamic/updated_skills_step90.json"
    PATH_7B_MODEL = "/home/bo.li/data/SkillRL/checkpoints/verl_agent_webshop/7b_hf_merged"
    PATH_7B_SKILL = "/home/bo.li/data/SkillRL/checkpoints/verl_agent_webshop/grpo_qwen2.5_7b_skills_dynamic/updated_skills_step153.json"

    print("🚀 [2/4] 正在加载 3B 和 7B 双模型...")
    llm_3b = LLM(model=PATH_3B_MODEL, tensor_parallel_size=1, gpu_memory_utilization=0.4, trust_remote_code=True)
    llm_7b = LLM(model=PATH_7B_MODEL, tensor_parallel_size=1, gpu_memory_utilization=0.4, trust_remote_code=True)

    sampling_params = SamplingParams(temperature=0.4, top_p=0.9, max_tokens=768, stop=["<|im_end|>"])

    print("🚀 [3/4] 正在加载你最新的真实训练 Config...")
    REAL_CONFIG_PATH = "/home/bo.li/SkillRL/outputs/2026-03-16/22-09-25/.hydra/config.yaml"
    base_config = OmegaConf.load(REAL_CONFIG_PATH)

    config_3b = base_config.copy()
    config_3b.env.use_skills_only_memory = True
    config_3b.env.skills_only_memory.skills_json_path = PATH_3B_SKILL
    config_3b.env.resources_per_worker = {'num_cpus': 1}

    config_7b = base_config.copy()
    config_7b.env.use_skills_only_memory = True
    config_7b.env.skills_only_memory.skills_json_path = PATH_7B_SKILL
    config_7b.env.resources_per_worker = {'num_cpus': 1}

    raw_env_3b = build_webshop_envs(seed=42, env_num=1, group_n=1, resources_per_worker={'num_cpus': 1}, is_train=False)
    raw_env_7b = build_webshop_envs(seed=42, env_num=1, group_n=1, resources_per_worker={'num_cpus': 1}, is_train=False)

    proj_func = partial(webshop_projection)
    mgr_3b = WebshopEnvironmentManager(raw_env_3b, proj_func, config_3b)
    mgr_7b = WebshopEnvironmentManager(raw_env_7b, proj_func, config_7b)

    print("🚀 [4/4] 正在调用 Manager 获取原生 Prompt...")
    obs_dict_3b, info_3b = mgr_3b.reset(kwargs={})
    obs_dict_7b, info_7b = mgr_7b.reset(kwargs={})

    task_desc = mgr_3b.tasks[0]
    print("=" * 80)
    print(f"🎯 竞技场目标任务: \033[92m{task_desc}\033[0m")
    print("=" * 80)

    state = {
        '3B': {'prompt_body': patch_step1_prompt(mgr_3b, obs_dict_3b['text'][0]), 'raw_obs': obs_dict_3b['anchor'][0],
               'done': False, 'reward': 0.0, 'step': 1, 'color': '\033[96m'},
        '7B': {'prompt_body': patch_step1_prompt(mgr_7b, obs_dict_7b['text'][0]), 'raw_obs': obs_dict_7b['anchor'][0],
               'done': False, 'reward': 0.0, 'step': 1, 'color': '\033[95m'}
    }

    while not (state['3B']['done'] and state['7B']['done']):
        for model_name, llm, mgr in [('3B', llm_3b, mgr_3b), ('7B', llm_7b, mgr_7b)]:
            s = state[model_name]
            if s['done']: continue

            c = s['color']

            # === 可视化展示 ===
            render_virtual_screen(model_name, s['raw_obs'], c)
            render_skills_panel(s['prompt_body'], c)

            qwen_prompt = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{s['prompt_body']}<|im_end|>\n<|im_start|>assistant\n"

            print(f"\n{c}[{model_name} - Step {s['step']}]\033[0m 🧠 正在思考...")
            outputs = llm.generate([qwen_prompt], sampling_params, use_tqdm=False)
            response_text = outputs[0].outputs[0].text

            think_match = re.search(r'<think>(.*?)</think>', response_text, re.IGNORECASE | re.DOTALL)
            if think_match:
                print(f"\033[90m{think_match.group(1).strip()}\033[0m")

            action_for_print = extract_action(response_text)
            print(f"{c}👉 {model_name} 意图执行: {action_for_print}\033[0m")

            next_obs_dict, reward_list, done_list, info_list = mgr.step([response_text])

            s['prompt_body'] = next_obs_dict['text'][0]
            s['raw_obs'] = next_obs_dict['anchor'][0]
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