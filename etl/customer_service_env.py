"""
Customer Service RL Environment Engine (Optimized for GRPO).

Implements a reinforcement learning environment based on playbook JSON scripts.
Features advanced reward shaping: Tiered Patience, Sentiment Delta, and Variance Smoothing.
"""

import json
import math
import random
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 31 valid skills (must match validator.py and llm_generator.py)
VALID_SKILLS = {
    # General skills
    'gen_greet', 'gen_empathize', 'gen_clarify', 'gen_verify_order',
    'gen_hold', 'gen_transfer', 'gen_apologize', 'gen_close',
    # Presale skills
    'pre_query_product', 'pre_check_stock', 'pre_compare', 'pre_recommend',
    'pre_answer_spec', 'pre_check_promo', 'pre_guide_purchase',
    # Logistics skills
    'log_query_status', 'log_query_detail', 'log_estimate_arrival',
    'log_modify_address', 'log_contact_courier', 'log_delay_notify', 'log_lost_claim',
    # Aftersale skills
    'aft_check_policy', 'aft_collect_evidence', 'aft_initiate_refund',
    'aft_initiate_return', 'aft_initiate_exchange', 'aft_schedule_pickup',
    'aft_track_progress', 'aft_compensate', 'aft_reject_explain'
}

# High-risk skills that require order validation
HIGH_RISK_SKILLS = {
    'aft_initiate_refund', 'aft_initiate_return',
    'aft_initiate_exchange', 'aft_compensate'
}

# 安全的沟通与兜底动作（不推进业务，但不会激怒买家）
SAFE_FALLBACK_SKILLS = {
    'gen_clarify',    # 澄清意图
    'gen_empathize',  # 安抚共情
    'gen_greet',      # 寒暄打招呼
    'gen_apologize',  # 道歉
    'gen_hold'        # 请稍等
}

# 情绪量化分值（用于计算 Sentiment Delta）
SENTIMENT_SCORES = {
    'happy': 1.0,
    'neutral': 0.0,
    'frustrated': -1.0,
    'angry': -2.0
}


@dataclass
class EnvState:
    """Environment state tracking."""
    current_node: str = 'root'
    action_history: List[str] = field(default_factory=list)
    dialogue_history: List[Dict[str, str]] = field(default_factory=list)
    visited_nodes: List[str] = field(default_factory=list)
    slots: Dict[str, Any] = field(default_factory=dict)
    done: bool = False
    won: bool = False
    fell_back: bool = False
    scenario: str = 'unknown'
    patience: int = 2  # 客户耐心值，初始允许犯错 2 次
    safe_action_consecutive_count: int = 0  # 连续使用安全话术的次数


class CustomerServiceEnv:
    """
    RL Environment for Customer Service Agent Training.
    Loads playbooks and provides step/episode rewards based on business outcomes.
    """

    def __init__(self, playbook_path: str):
        self.playbooks: List[Dict[str, Any]] = []
        self.current_playbook: Optional[Dict[str, Any]] = None
        self.state: Optional[EnvState] = None

        self._load_playbooks(playbook_path)
        logger.info(f"[CustomerServiceEnv] Loaded {len(self.playbooks)} playbooks")

    def _load_playbooks(self, path: str) -> None:
        with open(path, 'r', encoding='utf-8') as f:
            self.playbooks = json.load(f)
        if not self.playbooks:
            raise ValueError(f"No playbooks found in {path}")

    def reset(self, playbook_id: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if playbook_id:
            self.current_playbook = next(
                (p for p in self.playbooks if p['playbook_id'] == playbook_id), None
            )
            if not self.current_playbook:
                raise ValueError(f"Playbook not found: {playbook_id}")
        else:
            self.current_playbook = random.choice(self.playbooks)

        self.state = EnvState(
            current_node='root',
            action_history=[],
            dialogue_history=[],
            visited_nodes=['root'],
            slots=self.current_playbook.get('initial_slots', {}).copy(),
            done=False,
            won=False,
            fell_back=False,
            scenario=self.current_playbook.get('scenario', 'unknown'),
            patience=2,
            safe_action_consecutive_count=0
        )

        observation = self._get_observation()
        info = {
            'playbook_id': self.current_playbook['playbook_id'],
            'session_id': self.current_playbook.get('session_id'),
            'scenario': self.state.scenario,
            'business_outcome': self.current_playbook.get('business_outcome', {})
        }
        return observation, info

    def _get_observation(self) -> Dict[str, Any]:
        if not self.current_playbook or not self.state:
            return {}

        nodes = self.current_playbook['nodes']
        current_node_data = nodes.get(self.state.current_node, {})

        return {
            'node_id': self.state.current_node,
            'node': current_node_data,
            'buyer_text': current_node_data.get('buyer_text', ''),
            'sentiment': current_node_data.get('sentiment', 'neutral'),
            'available_skills': current_node_data.get('available_skills', []),
            'slots': self.state.slots,
            'action_history': self.state.action_history.copy(),
            'dialogue_history': self.state.dialogue_history.copy(),
            'done': self.state.done,
            'scenario': self.state.scenario,
            'patience': self.state.patience
        }

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self.state:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self.state.done:
            return self._get_observation(), 0.0, True, {'info': 'Episode already done'}

        nodes = self.current_playbook['nodes']
        current_node_data = nodes.get(self.state.current_node, {})
        current_sentiment = current_node_data.get('sentiment', 'neutral')

        # 记录买家当前消息
        current_buyer_text = current_node_data.get('buyer_text', '')
        if current_buyer_text and current_buyer_text != '[END]':
            self.state.dialogue_history.append({'role': 'buyer', 'content': current_buyer_text})

        # 记录模型动作
        self.state.action_history.append(action)
        self.state.dialogue_history.append({
            'role': 'agent',
            'action': action,
            'content': f"[Action: {action}]"
        })

        # ================= 模块一：步级奖励重塑 =================
        step_reward = -0.05  # 全局时间惩罚，鼓励高效
        self.state.fell_back = False
        transitions = current_node_data.get('transitions', {})
        next_node = self.state.current_node  # 默认留在原地（兜底状态）

        # ================= 模块二：分层容错与耐心系统 =================
        if action not in VALID_SKILLS:
            # Tier 1: 格式崩溃 (Syntax Error)
            self.state.fell_back = True
            self.state.patience -= 1
            step_reward -= 1.0
            self.state.dialogue_history.append({
                'role': 'system',
                'content': '[System: 动作格式无效，只能从 available_skills 中选择]'
            })
            logger.debug(f"Tier 1 格式错误: {action}, 剩余耐心: {self.state.patience}")

        elif action not in transitions:
            # Tier 2: 动作没命中正确路径，根据其语义属性来判定惩罚力度

            if action in SAFE_FALLBACK_SKILLS:
                # 【宽容判定：安全话术】
                self.state.fell_back = True
                self.state.safe_action_consecutive_count += 1

                # 动态递增惩罚机制：防滥用
                if self.state.safe_action_consecutive_count <= 2:
                    # 前两次：正常宽容，不扣耐心，轻微拖延惩罚
                    step_reward -= 0.1
                    msg = f"【系统判定】: 动作 [{action}] 是安全的沟通话术，但未能推进核心业务。请根据上下文尽快切入正题。"
                elif self.state.safe_action_consecutive_count <= 4:
                    # 3-4次：警告，加重惩罚，依然不扣耐心
                    step_reward -= 0.3
                    msg = f"【系统判定】: 警告！连续 {self.state.safe_action_consecutive_count} 次使用沟通话术，已造成对话拖延。请立即执行有效业务动作。"
                else:
                    # 5次以上：买家被激怒，严厉打击
                    self.state.patience -= 1
                    step_reward -= 0.5
                    msg = f"【系统判定】: 严重警告！过度滥用兜底话术 [{action}] 已引起买家反感（耐心-1）。必须立即选择业务动作推进流程！"

                self.state.dialogue_history.append({'role': 'system', 'content': msg})
                logger.debug(f"兜底动作: {action}, 连续次数: {self.state.safe_action_consecutive_count}, 剩余耐心: {self.state.patience}")

            else:
                # 【致命打击：业务偏航】
                # 重置安全计数器，因为模型至少"尝试"了去做业务动作
                self.state.safe_action_consecutive_count = 0
                self.state.fell_back = True
                self.state.patience -= 1
                step_reward -= 0.5

                self.state.dialogue_history.append({
                    'role': 'system',
                    'content': f"【系统判定】: 业务偏航！动作 [{action}] 在当前节点不可用。请重新分析并选择有效业务动作。"
                })
                logger.debug(f"Tier 2 业务偏航: {action}, 剩余耐心: {self.state.patience}")

        else:
            # 合法动作
            # 重置安全计数器，一旦选对动作，计数器就清零重置
            self.state.safe_action_consecutive_count = 0
            next_node = transitions[action]
            if next_node not in self.state.visited_nodes:
                step_reward += 0.5  # 防刷分：仅首次访问给分

            # 答对了，如果耐心受损，可恢复 1 点
            self.state.patience = min(2, self.state.patience + 1)

            if next_node == 'terminal':
                self.state.done = True
                self.state.won = True

        # Tier 3: 耐心耗尽 (Rage Quit)
        if self.state.patience <= 0 and not self.state.done:
            self.state.done = True
            self.state.won = False
            next_node = 'terminal'
            step_reward -= 2.0  # 严厉的愤怒挂断惩罚
            logger.info("[CustomerServiceEnv] Tier 3: 客户耐心耗尽，愤怒挂断！")

        # 更新节点状态
        self.state.current_node = next_node
        if next_node not in self.state.visited_nodes:
            self.state.visited_nodes.append(next_node)

        # ================= 模块三：情绪 Delta 引擎 =================
        if not self.state.fell_back and not self.state.done and next_node != 'terminal':
            next_sentiment = nodes.get(next_node, {}).get('sentiment', 'neutral')
            curr_score = SENTIMENT_SCORES.get(current_sentiment, 0.0)
            next_score = SENTIMENT_SCORES.get(next_sentiment, 0.0)
            delta = next_score - curr_score

            if delta > 0:
                step_reward += 0.8 * delta  # 成功安抚，强力正反馈
            elif delta < 0:
                step_reward += 1.0 * delta  # 激怒客户，严厉惩罚 (delta 是负数)

        # 更新槽位
        if not self.state.done and not self.state.fell_back:
            slot_updates = nodes.get(next_node, {}).get('slot_updates', {})
            self.state.slots.update(slot_updates)

        observation = self._get_observation()

        # 获取剧本真实的预设轮次（Ground Truth Length）
        target_steps = self.current_playbook.get('rl_steps', 0)

        info = {
            'fell_back': self.state.fell_back,
            'sentiment': nodes.get(next_node, {}).get('sentiment', 'neutral'),
            'visited_nodes': len(self.state.visited_nodes),
            'patience': self.state.patience,
            'won': self.state.won,
            'scenario': self.state.scenario,
            'business_outcome': self.current_playbook.get('business_outcome', {})
        }

        # 【修复：按剧本真实长度分桶记录成功率，避免幸存者偏差】
        # 使用 rl_steps（预设轮次）而非 current_steps（实际存活步数）进行分桶
        if self.state.done:
            if target_steps <= 5:
                info['success_rate/target_len_1_5'] = 1.0 if self.state.won else 0.0
            elif target_steps <= 10:
                info['success_rate/target_len_6_10'] = 1.0 if self.state.won else 0.0
            elif target_steps <= 15:
                info['success_rate/target_len_11_15'] = 1.0 if self.state.won else 0.0
            else:
                info['success_rate/target_len_16_20'] = 1.0 if self.state.won else 0.0

        return observation, step_reward, self.state.done, info

    def compute_episode_reward(self) -> float:
        """
        Compute final episode reward based on business outcome with Variance Smoothing.
        """
        if not self.state or not self.current_playbook:
            return 0.0

        business_outcome = self.current_playbook.get('business_outcome', {})
        has_order = business_outcome.get('has_order', False)
        order_amount = business_outcome.get('order_amount', 0.0)
        scenario = self.state.scenario
        won = self.state.won
        action_history = self.state.action_history

        reward = 0.0

        # ================= 模块四：终局平滑 =================
        # [Risk Control] 软化高危惩罚，防止模型产生“创伤后遗症”
        used_high_risk = any(skill in action_history for skill in HIGH_RISK_SKILLS)
        if not has_order and used_high_risk:
            logger.warning("[CustomerServiceEnv] Risk: High-risk action without order.")
            return -5.0  # 从 -10.0 平滑至 -5.0

        if scenario == "presale":
            if won:
                # 使用 log10 平滑极端金额，防止 Advantage 方差爆炸
                bonus = math.log10(max(10, order_amount)) * 0.5 if has_order else 0.0
                reward = 2.0 + bonus
            else:
                reward = -1.0

        elif scenario in ["logistics", "aftersale"]:
            if won:
                reward = 2.0
                if has_order and order_amount > 200:
                    reward += 1.0
            else:
                base_penalty = -2.0
                amount_penalty = -0.5 * math.log10(max(10, order_amount)) if has_order else 0.0
                reward = base_penalty + amount_penalty

        else:
            reward = 0.0 if won else -0.5

        # --- 终局情绪乘数 (Terminal Sentiment Multiplier) ---
        # 【修复漏洞2】：如果当前是 terminal，则回退 1 步取上一个真实的节点情绪
        if self.state.current_node == 'terminal' and len(self.state.visited_nodes) > 1:
            final_node_id = self.state.visited_nodes[-2]
        else:
            final_node_id = self.state.current_node

        final_node_data = self.current_playbook['nodes'].get(final_node_id, {})
        final_sentiment = final_node_data.get('sentiment', 'neutral')

        # 【修复漏洞1】：正负分数分离乘法，修复逻辑反转陷阱
        if final_sentiment == 'happy':
            reward = reward * 1.2 if reward > 0 else reward * 0.8  # 奖励放大，惩罚减小
        elif final_sentiment == 'angry':
            reward = reward * 0.8 if reward > 0 else reward * 1.2  # 奖励打折，惩罚加重

        logger.info(
            f"[CustomerServiceEnv] Episode reward: {reward:.2f} "
            f"(scenario={scenario}, won={won}, final_sentiment={final_sentiment})"
        )

        return reward

    def get_available_actions(self) -> List[str]:
        if not self.state or not self.current_playbook:
            return []
        nodes = self.current_playbook['nodes']
        current_node_data = nodes.get(self.state.current_node, {})
        return current_node_data.get('available_skills', list(VALID_SKILLS))

    def render(self) -> str:
        if not self.state:
            return "Environment not initialized"
        nodes = self.current_playbook['nodes']
        node = nodes.get(self.state.current_node, {})
        lines = [
            f"=== CustomerServiceEnv ===",
            f"Playbook: {self.current_playbook['playbook_id']}",
            f"Scenario: {self.state.scenario}",
            f"Current Node: {self.state.current_node}",
            f"Patience: {self.state.patience}/2",
            f"Buyer: {node.get('buyer_text', '')[:50]}...",
            f"Sentiment: {node.get('sentiment', 'neutral')}",
            f"Actions: {self.state.action_history}",
            f"Done: {self.state.done}, Won: {self.state.won}",
            "",
            "=== Dialogue History ===",
        ]
        for turn in self.state.dialogue_history:
            if turn['role'] == 'buyer':
                lines.append(f"[买家] {turn['content'][:80]}...")
            elif turn['role'] == 'system':
                lines.append(f"{turn['content']}")
            else:
                lines.append(f"[客服] {turn.get('content', turn.get('action', 'N/A'))}")
        return "\n".join(lines)


def create_env(playbook_path: str = "outputs/playbooks.json") -> CustomerServiceEnv:
    return CustomerServiceEnv(playbook_path)


def run_random_episode(env: CustomerServiceEnv) -> Dict[str, Any]:
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    while not env.state.done and steps < 20:
        available = env.get_available_actions()
        action = random.choice(available) if available else 'gen_clarify'
        obs, reward, done, step_info = env.step(action)
        total_reward += reward
        steps += 1

    episode_reward = env.compute_episode_reward()
    total_reward += episode_reward

    return {
        'playbook_id': info['playbook_id'],
        'scenario': info['scenario'],
        'steps': steps,
        'total_reward': total_reward,
        'episode_reward': episode_reward,
        'won': env.state.won,
        'business_outcome': info['business_outcome']
    }