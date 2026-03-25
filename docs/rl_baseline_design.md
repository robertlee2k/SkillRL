# RL Baseline 对接设计规划

> **文档版本**: v1.0
> **日期**: 2026-03-25
> **状态**: 待 Review，禁止直接编码

---

## 目录
1. [现状总结](#一现状总结)
2. [模块1：Gym Wrapper 与 Prompt Builder](#二模块1gym-wrapper-与-prompt-builder)
3. [模块2：Action Projection 与容错拦截](#三模块2action-projection-与容错拦截)
4. [模块3：RL Loop 与 Reward 收集机制](#四模块3rl-loop-与-reward-收集机制)
5. [模块4：监控指标](#五模块4监控指标)
6. [工程建议与风险控制](#六工程建议与风险控制)

---

## 一、现状总结

### 1.1 核心资产清单

| 文件 | 职责 | 关键要素 |
|------|------|----------|
| `playbooks_new.json` | 剧本库 | 树状结构、`business_outcome`、`sentiment`、`default_fallback` |
| `customer_service_env.py` | RL 环境 | `step()` 流转、`compute_episode_reward()` 阶梯算分 |
| `validator.py` | 动作校验 | 31 个离散 Skill 白名单 |
| `config.py` | 配置中心 | Skill 定义、场景分类 |

### 1.2 关键常量

```
VALID_SKILLS: 31 个离散 Skill（无参数）
HIGH_RISK_SKILLS: ['aft_initiate_refund', 'aft_initiate_return',
                   'aft_initiate_exchange', 'aft_compensate']
MAX_STEPS: 20 (建议值)
```

### 1.3 奖励机制摘要

| 类型 | 触发条件 | 奖励值 |
|------|----------|--------|
| **Step Reward - 推进** | 动作命中 transitions | +0.5 |
| **Step Reward - Fallback** | 动作未命中，触发 default_fallback | -0.5 |
| **Step Reward - 负向情绪** | 进入 `sentiment='angry'` 节点 | -0.5 |
| **Episode - 售前成单** | `won=True` + `has_order=True` | 2.0 + 0.01 × order_amount |
| **Episode - 售前流失** | `won=False` | -1.0 |
| **Episode - 物流/售后成功** | `won=True` | 2.0 (+1.0 if order_amount > 200) |
| **Episode - 物流/售后失败** | `won=False` | -2.0 - 0.5 × log₁₀(order_amount) |
| **风控红线** | 高危 Skill + `has_order=False` | **-10.0** (致命惩罚) |

---

## 二、模块1：Gym Wrapper 与 Prompt Builder

### 2.1 设计目标

将 `CustomerServiceEnv` 的内部状态转换为 LLM 可理解的 Prompt，并提供标准 Gym 接口供 RL 框架调用。

### 2.2 类设计

```python
class CustomerServiceGymWrapper(gym.Env):
    """
    将 CustomerServiceEnv 包装为标准 Gym 接口。

    职责：
    1. 调用底层 env 的 reset/step
    2. 将 observation 转换为 text prompt
    3. 解析 LLM 输出的 action
    4. 处理非法动作
    """

    def __init__(self, playbook_path: str, max_steps: int = 20):
        self.base_env = CustomerServiceEnv(playbook_path)
        self.max_steps = max_steps
        self.step_count = 0

        # 动作空间：31 个离散 Skill
        self.action_space = gym.spaces.Discrete(31)
        self.skill_id_to_idx = {skill: i for i, skill in enumerate(VALID_SKILLS)}
        self.idx_to_skill_id = {i: skill for skill, i in self.skill_id_to_idx.items()}

    def reset(self, **kwargs) -> Tuple[str, Dict]:
        """返回初始 Prompt 和 info"""
        obs, info = self.base_env.reset(**kwargs)
        self.step_count = 0
        prompt = self._build_prompt(obs)
        return prompt, info

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """执行动作，返回新 Prompt、reward、done、info"""
        # ... (详见 2.4)
```

### 2.3 Prompt Builder 设计

#### 2.3.1 Prompt 模板草案

```
你是一名专业的电商客服智能助手。你正在与买家进行对话，需要选择正确的服务动作。

## 当前对话状态

**场景类型**: {scenario}
**当前节点**: {node_id}
**买家情绪**: {sentiment}

**买家最新消息**:
{buyer_text}

**已执行动作序列**:
{action_history_formatted}

**当前槽位状态**:
{slots_formatted}

## 可用动作列表

以下是你当前可以选择的动作（请只选择一个）：

{available_skills_formatted}

## 动作说明

{skill_descriptions}

## 输出格式要求

请先分析买家意图和当前对话状态，然后选择最合适的动作。

<think>
[你的分析过程：买家想要什么？当前对话处于什么阶段？哪个动作最合适？]
</think>

<action>skill_id</action>

请确保 <action> 标签中的 skill_id 严格来自上述可用动作列表。
```

#### 2.3.2 Prompt Builder 实现

```python
class PromptBuilder:
    """构建发给 LLM 的 Prompt"""

    SYSTEM_PROMPT = """你是一名专业的电商客服智能助手，擅长处理售前咨询、物流查询和售后问题。
你的目标是高效解决买家问题，促成交易或妥善处理售后，同时避免激怒买家。"""

    @staticmethod
    def build(observation: Dict[str, Any]) -> str:
        """将环境 observation 转换为 Prompt"""
        node = observation.get('node', {})

        # 格式化可用技能
        available_skills = observation.get('available_skills', [])
        skills_text = "\n".join([
            f"- `{skill}`: {SKILL_DEFINITIONS.get(skill, {}).get('description', '未知')}"
            for skill in available_skills
        ])

        # 格式化动作历史
        history = observation.get('action_history', [])
        history_text = " → ".join(history) if history else "（无历史动作）"

        # 格式化槽位
        slots = observation.get('slots', {})
        slots_text = "\n".join([f"- {k}: {v}" for k, v in slots.items()]) if slots else "（暂无槽位信息）"

        prompt = f"""## 当前对话状态

**场景类型**: {observation.get('scenario', 'unknown')}
**当前节点**: {observation.get('node_id', 'root')}
**买家情绪**: {observation.get('sentiment', 'neutral')}

**买家最新消息**:
{observation.get('buyer_text', '')}

**已执行动作序列**:
{history_text}

**当前槽位状态**:
{slots_text}

## 可用动作列表

以下是你当前可以选择的动作：

{skills_text}

## 输出格式要求

先分析买家意图和对话状态，再选择动作：

<think>
[分析：买家想要什么？对话处于什么阶段？哪个动作最合适？]
</think>

<action>skill_id</action>"""

        return prompt
```

### 2.4 Observation 格式转换

```python
def _get_observation(self) -> Dict[str, Any]:
    """获取格式化的观察"""
    base_obs = self.base_env._get_observation()

    # 添加额外信息
    base_obs['scenario'] = self.base_env.state.scenario
    base_obs['step_count'] = self.step_count
    base_obs['prompt'] = PromptBuilder.build(base_obs)

    return base_obs
```

---

## 三、模块2：Action Projection 与容错拦截

### 3.1 设计目标

从 LLM 的 CoT 输出中可靠提取 `skill_id`，并对非法动作进行拦截和修正。

### 3.2 LLM 输出格式约定

```
<think>
买家发送了商品链接，询问重力球奶嘴是哪一个。
这是一个售前咨询场景，买家有明确的产品问题。
当前节点是 root，买家情绪 calm。
我应该先回答产品规格问题，选择 pre_answer_spec。
</think>

<action>pre_answer_spec</action>
```

### 3.3 正则解析逻辑

```python
import re

class ActionParser:
    """解析 LLM 输出的 Action"""

    # 正则模式（按优先级排序）
    PATTERNS = [
        # 标准格式: <action>skill_id</action>
        r'<action>\s*([a-z_]+)\s*</action>',
        # 宽松格式: action: skill_id 或 动作: skill_id
        r'(?:action|动作)\s*[:：]\s*([a-z_]+)',
        # 直接提取 skill_id 模式（保底）
        r'\b(gen_[a-z_]+|pre_[a-z_]+|log_[a-z_]+|aft_[a-z_]+)\b',
    ]

    @classmethod
    def parse(cls, llm_output: str) -> Optional[str]:
        """
        从 LLM 输出中提取 skill_id。

        Returns:
            skill_id 如果找到且合法，否则 None
        """
        llm_output = llm_output.strip()

        for pattern in cls.PATTERNS:
            match = re.search(pattern, llm_output, re.IGNORECASE)
            if match:
                skill_id = match.group(1).lower()
                # 验证是否在白名单
                if skill_id in VALID_SKILLS:
                    return skill_id

        return None
```

### 3.4 容错拦截设计

#### 3.4.1 异常分类与处理策略

| 异常类型 | 示例 | 处理策略 |
|----------|------|----------|
| **带参数的 Skill** | `aft_initiate_refund[order=123]` | 提取前缀 → 验证白名单 → 若合法则使用，否则 fallback |
| **非法 Skill ID** | `do_refund` / `unknown_action` | 返回 fallback action |
| **格式完全崩坏** | 纯乱码或无 action 标签 | 返回 fallback action |
| **空输出** | `""` | 返回 fallback action |
| **多个 Action** | `<action>a</action><action>b</action>` | 取第一个 |

#### 3.4.2 Fallback Action 选择策略

```python
class ActionInterceptor:
    """动作拦截与修正"""

    def __init__(self, fallback_strategy: str = 'clarify'):
        self.fallback_strategy = fallback_strategy

    def intercept(
        self,
        raw_action: str,
        available_skills: List[str],
        observation: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        拦截并修正非法动作。

        Returns:
            (修正后的 action, 拦截信息 dict)
        """
        parse_result = ActionParser.parse(raw_action)
        intercept_info = {
            'original_output': raw_action,
            'parsed_action': parse_result,
            'was_intercepted': False,
            'intercept_reason': None
        }

        # Case 1: 解析成功且在可用列表中
        if parse_result and parse_result in available_skills:
            return parse_result, intercept_info

        # Case 2: 解析成功但不在可用列表（尝试模糊匹配）
        if parse_result and parse_result in VALID_SKILLS:
            intercept_info['was_intercepted'] = True
            intercept_info['intercept_reason'] = 'skill_not_available'
            # 选择语义最接近的可用 skill 或 fallback
            return self._select_fallback(available_skills, observation), intercept_info

        # Case 3: 解析失败
        intercept_info['was_intercepted'] = True
        intercept_info['intercept_reason'] = 'parse_failed' if not parse_result else 'invalid_skill'

        return self._select_fallback(available_skills, observation), intercept_info

    def _select_fallback(
        self,
        available_skills: List[str],
        observation: Dict[str, Any]
    ) -> str:
        """
        选择 fallback action。

        策略优先级：
        1. 如果有 gen_clarify 可用 → 返回 gen_clarify（澄清意图）
        2. 如果有 gen_empathize 可用 → 返回 gen_empathize（安抚情绪）
        3. 如果有 gen_transfer 可用 → 返回 gen_transfer（转人工，安全兜底）
        4. 返回 available_skills[0]（强行选择第一个）
        """
        priority = ['gen_clarify', 'gen_empathize', 'gen_transfer']

        for skill in priority:
            if skill in available_skills:
                return skill

        return available_skills[0] if available_skills else 'gen_clarify'
```

### 3.5 带参数 Skill 的特殊处理

```python
def extract_skill_without_params(raw_action: str) -> Optional[str]:
    """
    从带参数的 Skill 中提取基础 Skill ID。

    Examples:
        'aft_initiate_refund[order=123]' → 'aft_initiate_refund'
        'pre_query_product[sku=456]' → 'pre_query_product'
    """
    # 移除方括号及其内容
    clean = re.sub(r'\[.*?\]', '', raw_action).strip()
    if clean in VALID_SKILLS:
        return clean
    return None
```

---

## 四、模块3：RL Loop 与 Reward 收集机制

### 4.1 框架选型建议

| 框架 | 优势 | 劣势 | 推荐度 |
|------|------|------|--------|
| **TRL** | 与 HuggingFace 生态集成好，API 简洁 | 大规模分布式支持弱 | ⭐⭐⭐⭐ (推荐起步) |
| **veRL** | 高性能，支持大规模分布式 | 学习曲线陡峭 | ⭐⭐⭐ (推荐后期) |
| **Ray RLlib** | 成熟的分布式框架 | 对 LLM 支持不如前两者 | ⭐⭐ |

**推荐路径**: 先用 TRL 跑通 Pipeline，验证收敛后再迁移至 veRL 优化性能。

### 4.2 训练主循环设计

```python
class RLPipeline:
    """RL 训练 Pipeline"""

    def __init__(
        self,
        policy_model,          # 待训练的 LLM (e.g., Qwen)
        ref_model,             # 参考模型 (frozen, 用于 KL penalty)
        env: CustomerServiceGymWrapper,
        tokenizer,
        config: RLConfig
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.env = env
        self.tokenizer = tokenizer
        self.config = config

    def collect_rollouts(self, num_episodes: int) -> RolloutBuffer:
        """
        收集 Rollout 数据。

        每个 episode:
        1. reset env → 获取初始 prompt
        2. 循环 max_steps 次:
           a. policy_model 生成输出（含 CoT + action）
           b. 解析 action → 拦截修正
           c. env.step(action) → 获取 step_reward
           d. 记录 (prompt, output, step_reward, done)
        3. episode 结束时调用 compute_episode_reward()
        """
        buffer = RolloutBuffer()

        for _ in range(num_episodes):
            prompt, info = self.env.reset()
            episode_rewards = []
            episode_logits = []
            episode_values = []

            for step in range(self.config.max_steps):
                # LLM 生成
                output = self._generate(prompt)
                action, intercept_info = self._parse_action(output)

                # 环境交互
                next_prompt, step_reward, done, step_info = self.env.step(action)

                # 记录
                episode_rewards.append(step_reward)
                buffer.add(
                    prompt=prompt,
                    output=output,
                    reward=step_reward,
                    done=done,
                    intercept_info=intercept_info
                )

                prompt = next_prompt
                if done:
                    break

            # Episode 结束，计算最终奖励
            episode_reward = self.env.base_env.compute_episode_reward()

            # 将 episode reward 分配到各 step (策略：均匀分配或末尾集中)
            self._allocate_episode_reward(buffer, episode_reward, len(episode_rewards))

        return buffer

    def _allocate_episode_reward(
        self,
        buffer: RolloutBuffer,
        episode_reward: float,
        num_steps: int
    ):
        """
        将 Episode Reward 分配到各 Step。

        策略：
        - 末尾集中：episode_reward 全部加到最后一步
        - 或者：按 gamma 衰减分配
        """
        # 末尾集中策略（简单，推荐初期使用）
        buffer.rewards[-1] += episode_reward
```

### 4.3 Advantage 计算

```python
def compute_advantages(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.99,
    lam: float = 0.95
) -> Tuple[List[float], List[float]]:
    """
    使用 GAE (Generalized Advantage Estimation) 计算优势函数。

    A_t = Σ (γλ)^l * δ_{t+l}

    其中 δ_t = r_t + γ * V_{t+1} - V_t
    """
    advantages = []
    returns = []
    gae = 0.0

    # 从后向前计算
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae

        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])

    # Normalize advantages (稳定训练)
    advantages = torch.tensor(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages.tolist(), returns
```

### 4.4 PPO Loss 设计

```python
def compute_ppo_loss(
    policy_model,
    ref_model,
    prompts: List[str],
    outputs: List[str],
    advantages: List[float],
    clip_ratio: float = 0.2,
    kl_coef: float = 0.1
) -> torch.Tensor:
    """
    PPO Loss = Policy Loss + Value Loss + KL Penalty

    Policy Loss (Clipped):
        L^{CLIP} = -E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]

    KL Penalty:
        L^{KL} = KL(π_θ || π_ref)

    Value Loss:
        L^{VF} = MSE(V_θ(s), R)
    """
    # 计算当前 policy 的 log probs
    policy_logprobs = compute_logprobs(policy_model, prompts, outputs)
    ref_logprobs = compute_logprobs(ref_model, prompts, outputs)

    # 计算 ratio
    log_ratio = policy_logprobs - ref_logprobs.detach()
    ratio = torch.exp(log_ratio)

    # Clipped policy loss
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    # KL penalty
    kl_loss = (policy_logprobs - ref_logprobs).mean().abs()

    # Total loss
    total_loss = policy_loss + kl_coef * kl_loss

    return total_loss, {
        'policy_loss': policy_loss.item(),
        'kl_loss': kl_loss.item(),
        'kl_coef': kl_coef
    }
```

### 4.5 超参数初始建议

```python
@dataclass
class RLConfig:
    """RL 训练超参数"""

    # Episode 设置
    max_steps: int = 20                    # 每个 episode 最大步数
    num_episodes_per_batch: int = 64       # 每批次收集的 episode 数

    # PPO 参数
    clip_ratio: float = 0.2                # PPO clip 范围
    gamma: float = 0.99                    # 折扣因子
    lam: float = 0.95                      # GAE lambda

    # KL 控制
    kl_coef: float = 0.1                   # KL 惩罚系数
    kl_target: float = 0.1                 # 目标 KL 散度
    kl_horizon: int = 100                  # KL 自适应调整周期

    # 学习率
    learning_rate: float = 1e-6            # 策略网络学习率（小模型可用 1e-5）
    lr_scheduler: str = 'cosine'           # 学习率调度

    # Batch 设置
    batch_size: int = 8                    # Mini-batch 大小
    gradient_accumulation_steps: int = 4   # 梯度累积步数

    # 训练轮次
    ppo_epochs: int = 4                    # 每批数据的 PPO 更新轮数
    max_train_steps: int = 10000           # 最大训练步数

    # 生成参数
    max_new_tokens: int = 256              # 最大生成 token 数
    temperature: float = 1.0               # 采样温度
    top_p: float = 0.9                     # Nucleus sampling
```

### 4.6 防止模式崩塌的工程措施

| 风险 | 原因 | 解决方案 |
|------|------|----------|
| **模式崩塌** | 早期 reward 信号弱，模型收敛到单一动作 | 1. 使用 KL Penalty 限制偏离参考模型<br>2. 初始化使用 SFT 模型<br>3. 初期使用较大的 entropy bonus |
| **Reward Hacking** | 模型学会欺骗奖励函数 | 1. 多样化训练数据（不同 scenario）<br>2. 监控 Valid Action Rate |
| **Catastrophic Forgetting** | RL 训练覆盖预训练知识 | 1. 较小的学习率 (1e-6)<br>2. 定期评估通用能力 |
| **KL 爆炸** | Policy 偏离 Ref 过远 | 1. 自适应 KL 系数<br>2. 早停机制 |

---

## 五、模块4：监控指标

### 5.1 核心训练指标

| 指标名 | 计算方式 | 意义 | 告警阈值 |
|--------|----------|------|----------|
| `mean_episode_reward` | episode 总 reward 平均值 | 训练效果核心指标 | 持续低于 -5 |
| `mean_step_reward` | step reward 平均值 | 单步决策质量 | 持续低于 0 |
| `win_rate` | `won=True` 的 episode 比例 | 成功解决对话比例 | 持续低于 20% |
| `policy_loss` | PPO policy loss | 收敛状态 | 震荡不下降 |
| `kl_divergence` | KL(π_θ \|\| π_ref) | 偏离参考模型程度 | > 1.0 |
| `entropy` | 动作分布熵 | 探索程度 | < 0.5 |

### 5.2 动作质量指标

| 指标名 | 计算方式 | 意义 | 告警阈值 |
|--------|----------|------|----------|
| **`valid_action_rate`** | `解析成功且合法的 action / 总 action` | LLM 输出质量 | < 80% |
| **`fallback_rate`** | `触发 fallback 的 step / 总 step` | 决策错误率 | > 30% |
| **`intercept_rate`** | `被拦截修正的 action / 总 action` | 容错触发频率 | > 20% |
| **`parse_fail_rate`** | `解析失败的 action / 总 action` | LLM 格式遵循度 | > 10% |
| `action_diversity` | episode 内不同 action 数量 | 模式崩塌检测 | 始终为 1-2 |

### 5.3 业务指标

| 指标名 | 计算方式 | 意义 | 目标 |
|--------|----------|------|------|
| **`high_risk_violation_rate`** | `高危动作 + has_order=False / 总高危动作` | 风控红线触犯率 | **= 0%** |
| `presale_conversion_rate` | 售前场景 `won=True` 比例 | 售前转化率 | > 50% |
| `aftersale_resolution_rate` | 售后场景 `won=True` 比例 | 售后解决率 | > 60% |
| `avg_order_amount` | 成功 episode 的平均客单价 | 高价值客户捕获 | 监控趋势 |

### 5.4 情绪轨迹指标

| 指标名 | 计算方式 | 意义 |
|--------|----------|------|
| `angry_node_visit_rate` | 进入 `sentiment='angry'` 节点的比例 | 买家情绪恶化检测 |
| `sentiment_trajectory` | episode 内 sentiment 变化序列 | 对话质量分析 |

### 5.5 日志格式规范

```python
import wandb

class MetricsLogger:
    """训练指标记录器"""

    def log_episode(
        self,
        episode_info: Dict[str, Any],
        step: int
    ):
        """记录单个 episode 的指标"""
        metrics = {
            # 训练指标
            'train/episode_reward': episode_info['total_reward'],
            'train/episode_length': episode_info['num_steps'],
            'train/win_rate': 1.0 if episode_info['won'] else 0.0,

            # 动作质量
            'action/valid_action_rate': episode_info['valid_action_count'] / episode_info['num_steps'],
            'action/fallback_rate': episode_info['fallback_count'] / episode_info['num_steps'],
            'action/intercept_rate': episode_info['intercept_count'] / episode_info['num_steps'],
            'action/diversity': len(set(episode_info['action_history'])),

            # 业务指标
            'business/has_order': episode_info['business_outcome'].get('has_order', False),
            'business/order_amount': episode_info['business_outcome'].get('order_amount', 0),
            'business/high_risk_violation': episode_info.get('high_risk_violation', False),

            # 情绪
            'sentiment/angry_visits': episode_info.get('angry_node_visits', 0),
        }

        wandb.log(metrics, step=step)
```

### 5.6 监控仪表板建议

```
┌─────────────────────────────────────────────────────────────┐
│                    RL Training Dashboard                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │ Episode Reward  │  │ Win Rate        │  │ KL Div      │  │
│  │   ▁▂▃▄▅▆▇█     │  │   ▁▂▃▄▅▆       │  │   ▁▂▃▄      │  │
│  │   Mean: 1.23    │  │   58.3%         │  │   0.08      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │ Valid Action %  │  │ Fallback Rate   │  │ Violation   │  │
│  │   ████████░░    │  │   ████░░░░░░    │  │   ▓▓░░░░    │  │
│  │   92.1%         │  │   12.5%         │  │   0.0% ✓    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Action Distribution (Last 1000 steps)                       │
│  pre_answer_spec: ████████████████ 28%                       │
│  pre_recommend:   ████████████ 22%                           │
│  gen_clarify:     ████████ 15%                               │
│  gen_empathize:   ██████ 12%                                 │
│  ...                                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 六、工程建议与风险控制

### 6.1 分阶段实施路线

```
Phase 1: 跑通 Pipeline (1-2 周)
├── 实现 GymWrapper + PromptBuilder
├── 实现 ActionParser + ActionInterceptor
├── 用 TRL + Qwen-2.5-3B 跑通最小训练循环
├── 验证 reward 曲线有上升趋势
└── 交付物: 能跑通的训练脚本

Phase 2: 稳定性优化 (1 周)
├── 调优超参数防止模式崩塌
├── 完善 Metrics 日志
├── 增加自动化测试
└── 交付物: 稳定收敛的 baseline

Phase 3: 性能提升 (1-2 周)
├── 尝试更大模型 (Qwen-2.5-7B)
├── 迁移至 veRL 提升训练效率
├── 探索 GRPO 算法替代 PPO
└── 交付物: 生产级训练 pipeline
```

### 6.2 关键风险与缓解措施

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| LLM 不遵循 action 格式 | 高 | 中 | 强化 Prompt 指令；SFT 预训练 |
| Reward 稀疏导致学习困难 | 中 | 高 | 增加 step reward 密度；Reward Shaping |
| 模式崩塌到单一动作 | 中 | 高 | KL Penalty；Entropy Bonus；Early Stopping |
| 风控红线被突破 | 低 | 致命 | 强制拦截高危动作；监控告警 |
| 训练数据不足 | 中 | 中 | 数据增强；合成数据 |

### 6.3 代码组织建议

```
skillrl/
├── etl/
│   ├── customer_service_env.py    # 现有环境
│   └── validator.py               # 现有校验
├── rl/                            # 新增目录
│   ├── __init__.py
│   ├── gym_wrapper.py             # GymWrapper + PromptBuilder
│   ├── action_parser.py           # ActionParser + ActionInterceptor
│   ├── pipeline.py                # RLPipeline
│   ├── config.py                  # RLConfig
│   ├── metrics.py                 # MetricsLogger
│   └── utils.py                   # 工具函数
├── scripts/
│   ├── train_rl.py                # 训练入口脚本
│   └── evaluate.py                # 评估脚本
└── tests/
    ├── test_gym_wrapper.py
    ├── test_action_parser.py
    └── test_pipeline.py
```

### 6.4 验收标准

| 检查项 | 标准 |
|--------|------|
| Pipeline 跑通 | 能完成 100 个 episode 训练不报错 |
| Reward 有上升趋势 | 1000 步后 mean_episode_reward > 初始值 |
| Valid Action Rate | > 80% |
| High-Risk Violation | = 0% |
| Win Rate | > 30% (初期基线) |

---

## 附录

### A. 31 个 Skill 完整列表

```
# General (8)
gen_greet, gen_empathize, gen_clarify, gen_verify_order,
gen_hold, gen_transfer, gen_apologize, gen_close

# Presale (7)
pre_query_product, pre_check_stock, pre_compare, pre_recommend,
pre_answer_spec, pre_check_promo, pre_guide_purchase

# Logistics (7)
log_query_status, log_query_detail, log_estimate_arrival,
log_modify_address, log_contact_courier, log_delay_notify, log_lost_claim

# Aftersale (9)
aft_check_policy, aft_collect_evidence, aft_initiate_refund,
aft_initiate_return, aft_initiate_exchange, aft_schedule_pickup,
aft_track_progress, aft_compensate, aft_reject_explain
```

### B. 参考 Issue 链接

- TRL Documentation: https://huggingface.co/docs/trl
- veRL Repository: https://github.com/volcengine/verl
- PPO Paper: https://arxiv.org/abs/1707.06347

---

**文档结束**

> 请 Review 此设计文档，确认无误后我们将进入代码实现阶段。