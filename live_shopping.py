#        skills_json_path="/home/bo.li/data/SkillRL/checkpoints/verl_agent_webshop/grpo_qwen2.5_3b_skills_dynamic/updated_skills_step90.json",

import os
import re
import ray
from vllm import LLM, SamplingParams
from agent_system.environments.env_package.webshop.envs import build_webshop_envs
from agent_system.memory.skills_only_memory import SkillsOnlyMemory


def extract_action(text: str) -> str:
    """提取大模型的动作指令"""
    match = re.search(r'<action>(.*?)</action>', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    fallback = re.search(r'(search\[.*?\]|click\[.*?\])', text, re.IGNORECASE)
    return fallback.group(1).strip() if fallback else "click[back to search]"


def main():
    print("🚀 [1/4] 正在初始化 Ray 引擎...")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    print("🚀 [2/4] 正在加载 RAG 动态技能库 (3B满血版)...")
    # 【注意】这里请确保填入你真实的 step90.json 路径！
    memory = SkillsOnlyMemory(
        skills_json_path="/home/bo.li/data/SkillRL/checkpoints/verl_agent_webshop/grpo_qwen2.5_3b_skills_dynamic/updated_skills_step90.json",
        # 请核对这个路径
        retrieval_mode="embedding",
        embedding_model_path="/home/bo.li/data/models/bge-small-en-v1.5",
        task_specific_top_k=6
    )

    print("🚀 [3/4] 正在加载合并后的 3B 模型 (vLLM)...")
    MODEL_PATH = "/home/bo.li/data/SkillRL/checkpoints/verl_agent_webshop/3b_hf_merged"
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=1, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=768, stop=["<|im_end|>"])

    print("🚀 [4/4] 正在初始化 WebShop 真实环境...")
    env = build_webshop_envs(
        seed=42,
        env_num=1,
        group_n=1,
        resources_per_worker={'num_cpus': 1},
        is_train=False
    )

    obs_list, info_list = env.reset()
    obs = obs_list[0]
    info = info_list[0]

    # ==========================================
    # 终极修复 1：基于 [SEP] 的完美任务提取
    # ==========================================
    task_desc = ""
    obs_str = str(obs)
    if "Instruction: [SEP]" in obs_str:
        # 切割后长这样: " Find me slip resistant... "
        task_part = obs_str.split("Instruction: [SEP]")[1]
        task_desc = task_part.split("[SEP]")[0].strip()

    if not task_desc:
        task_desc = "Search for a product (Fallback failed)"

    print("=" * 60)
    print(f"🎯 今天的购物任务是: \033[92m{task_desc}\033[0m")
    print("=" * 60)

    history_log = []
    step = 1

    while True:
        # 检索当前任务最需要的 Top-6 技能 (包含那 3 个关键的 dyn_ 护航技能)
        retrieved = memory.retrieve(task_desc, top_k=6)
        skills_text = memory.format_for_prompt(retrieved)
        history_str = "\n".join(history_log) if history_log else "No history yet."

        # ==========================================
        # 终极修复 2：精准解析 available_actions 字典
        # ==========================================
        raw_actions = info.get('available_actions', {})
        admissible_actions = []

        if isinstance(raw_actions, dict):
            if raw_actions.get('has_search_bar'):
                admissible_actions.append('search[<fill_in_your_keywords>]')
            for item in raw_actions.get('clickables', []):
                admissible_actions.append(f"click[{item}]")

        # 兜底：如果啥都没有，把整个 obs 传进去找括号
        if not admissible_actions:
            found_buttons = re.findall(r'\[(.*?)\]', str(obs))
            admissible_actions = [f"click[{btn.lower()}]" for btn in found_buttons if btn.lower() != 'search']
            if 'search' in str(obs).lower():
                admissible_actions.insert(0, 'search[<fill_in_your_keywords>]')

        actions_str = "\n".join([f"- {act}" for act in admissible_actions])

        prompt = f"""<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_desc}

## Retrieved Relevant Experience
{skills_text}

## Current Progress
{history_str}
You are now at step {step} and your current observation is: '{obs}'.
Your admissible actions of the current situation are:
{actions_str}

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.<|im_end|>
<|im_start|>assistant
"""
        print(f"\n[Step {step}] 🧠 模型正在思考...")
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        response_text = outputs[0].outputs[0].text

        think_match = re.search(r'<think>(.*?)</think>', response_text, re.IGNORECASE | re.DOTALL)
        if think_match:
            print(f"\033[90m{think_match.group(1).strip()}\033[0m")
        else:
            print(f"\033[90m(未捕捉到思考过程)\n{response_text[:100]}\033[0m")

        action = extract_action(response_text)
        print(f"👉 模型决定执行动作: \033[96m{action}\033[0m")

        history_log.append(f"[Observation {step}: '{obs[:100]}...', Action {step}: '{action}']")

        obs_list, reward_list, done_list, info_list = env.step([action])

        obs = obs_list[0]
        reward = reward_list[0]
        done = done_list[0]
        info = info_list[0]

        if done:
            print("=" * 60)
            print(f"🛒 购物结束！")
            print(f"💰 最终得分 (Reward): \033[93m{reward}\033[0m / 10.0")
            print("=" * 60)
            break

        step += 1
        if step > 15:
            print("❌ 超过最大步数，任务失败！")
            break


if __name__ == "__main__":
    main()