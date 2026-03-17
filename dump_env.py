import json
import ray
from agent_system.environments.env_package.webshop.envs import build_webshop_envs


def dump_env_state():
    print("🚀 [1/3] 正在初始化 Ray...")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    print("🚀 [2/3] 正在初始化 WebShop 环境...")
    env = build_webshop_envs(
        seed=42,
        env_num=1,
        group_n=1,
        resources_per_worker={'num_cpus': 1},
        is_train=False
    )

    print("🚀 [3/3] 正在调用 reset() 并捕获数据...")
    obs_list, info_list = env.reset()

    obs = obs_list[0]
    info = info_list[0]

    # 强行捕获环境对象上的所有可能包含任务的属性
    env_attrs = {}
    for attr in ['instruction_text', 'goal', 'task', 'prompt']:
        if hasattr(env, attr):
            env_attrs[attr] = str(getattr(env, attr))

    # 构造导出结构
    dump_data = {
        "1_obs_type": str(type(obs)),
        "2_obs_raw_content": str(obs),
        "3_info_type": str(type(info)),
        "4_info_keys": list(info.keys()) if isinstance(info, dict) else "Not a dict",
        "5_info_raw_content": str(info),
        "6_env_internal_attributes": env_attrs
    }

    # 保存为 JSON 文件
    output_file = "webshop_env_dump.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dump_data, f, indent=4, ensure_ascii=False)

    print("=" * 60)
    print(f"✅ 捕获成功！环境结构的底牌已全部保存至: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    dump_env_state()