"""
Customer Service RL Environment Engine.

Implements a reinforcement learning environment based on playbook JSON scripts.
Supports step rewards and episode-level business outcome rewards.
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


@dataclass
class EnvState:
    """Environment state tracking."""
    current_node: str = 'root'
    action_history: List[str] = field(default_factory=list)
    dialogue_history: List[Dict[str, str]] = field(default_factory=list)  # Full dialogue: [{"role": "buyer/agent", "content": "..."}]
    visited_nodes: List[str] = field(default_factory=list)
    slots: Dict[str, Any] = field(default_factory=dict)
    done: bool = False
    won: bool = False  # Successfully reached terminal
    fell_back: bool = False  # Whether last action triggered fallback
    scenario: str = 'unknown'


class CustomerServiceEnv:
    """
    RL Environment for Customer Service Agent Training.

    Loads playbooks and provides step/episode rewards based on business outcomes.
    """

    def __init__(self, playbook_path: str):
        """
        Initialize environment with playbooks.

        Args:
            playbook_path: Path to JSON file containing playbooks list
        """
        self.playbooks: List[Dict[str, Any]] = []
        self.current_playbook: Optional[Dict[str, Any]] = None
        self.state: Optional[EnvState] = None

        # Load playbooks
        self._load_playbooks(playbook_path)
        logger.info(f"[CustomerServiceEnv] Loaded {len(self.playbooks)} playbooks")

    def _load_playbooks(self, path: str) -> None:
        """Load playbooks from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            self.playbooks = json.load(f)

        if not self.playbooks:
            raise ValueError(f"No playbooks found in {path}")

    def reset(self, playbook_id: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            playbook_id: Optional specific playbook to use, else random

        Returns:
            Tuple of (observation, info) where observation contains:
                - current_node: node dict
                - slots: current slot values
                - scenario: scenario type
                - action_history: empty list
        """
        # Select playbook
        if playbook_id:
            self.current_playbook = next(
                (p for p in self.playbooks if p['playbook_id'] == playbook_id),
                None
            )
            if not self.current_playbook:
                raise ValueError(f"Playbook not found: {playbook_id}")
        else:
            self.current_playbook = random.choice(self.playbooks)

        # Initialize state
        self.state = EnvState(
            current_node='root',
            action_history=[],
            dialogue_history=[],
            visited_nodes=['root'],
            slots=self.current_playbook.get('initial_slots', {}).copy(),
            done=False,
            won=False,
            fell_back=False,
            scenario=self.current_playbook.get('scenario', 'unknown')
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
        """Get current observation."""
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
            'dialogue_history': self.state.dialogue_history.copy(),  # Full dialogue for LLM
            'done': self.state.done,
            'scenario': self.state.scenario
        }

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: Skill ID to execute

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if not self.state or self.state.done:
            raise RuntimeError("Environment not reset. Call reset() first.")

        if action not in VALID_SKILLS:
            logger.warning(f"[CustomerServiceEnv] Invalid action: {action}")
            return self._get_observation(), -1.0, True, {'error': 'Invalid action'}

        # Get current node data
        nodes = self.current_playbook['nodes']
        current_node_data = nodes.get(self.state.current_node, {})

        # Record current buyer message into dialogue history
        current_buyer_text = current_node_data.get('buyer_text', '')
        if current_buyer_text and current_buyer_text != '[END]':
            self.state.dialogue_history.append({
                'role': 'buyer',
                'content': current_buyer_text
            })

        # Track action
        self.state.action_history.append(action)

        # Record agent action into dialogue history
        self.state.dialogue_history.append({
            'role': 'agent',
            'action': action,
            'content': f"[Action: {action}]"  # In real scenario, LLM would generate response
        })

        # Calculate step reward
        step_reward = 0.0
        self.state.fell_back = False

        # Check if action is in transitions
        transitions = current_node_data.get('transitions', {})

        if action in transitions:
            # Valid transition
            next_node = transitions[action]
            step_reward += 0.5  # Successfully progressed

            # Check if reached terminal
            if next_node == 'terminal':
                self.state.done = True
                self.state.won = True
        else:
            # Fall back to default_fallback
            fallback = current_node_data.get('default_fallback', 'terminal')
            next_node = fallback
            self.state.fell_back = True
            step_reward -= 0.5  # Fallback penalty

        # Update state
        self.state.current_node = next_node
        self.state.visited_nodes.append(next_node)

        # Check sentiment for negative path penalty
        next_node_data = nodes.get(next_node, {})
        if next_node_data.get('sentiment') == 'angry':
            step_reward -= 0.5

        # Update slots
        slot_updates = next_node_data.get('slot_updates', {})
        self.state.slots.update(slot_updates)

        # Check if terminal
        if next_node == 'terminal':
            self.state.done = True

        observation = self._get_observation()
        info = {
            'fell_back': self.state.fell_back,
            'sentiment': next_node_data.get('sentiment', 'neutral'),
            'visited_nodes': len(self.state.visited_nodes)
        }

        return observation, step_reward, self.state.done, info

    def compute_episode_reward(self) -> float:
        """
        Compute final episode reward based on business outcome.

        This is called when episode terminates to calculate
        the overall success/failure reward.

        Returns:
            Final episode reward
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

        # [Risk Control] Check for high-risk actions without order
        used_high_risk = any(skill in action_history for skill in HIGH_RISK_SKILLS)

        if not has_order and used_high_risk:
            logger.warning("[CustomerServiceEnv] FATAL: High-risk action without order!")
            return -10.0  # Critical financial loss penalty

        if scenario == "presale":
            if won:
                # Successful presale conversion
                reward = 2.0 + (0.01 * order_amount) if has_order else 1.0
            else:
                # Presale churn penalty
                reward = -1.0

        elif scenario in ["logistics", "aftersale"]:
            if won:
                # Successfully resolved issue
                reward = 2.0
                # Bonus for high-value order recovery
                if has_order and order_amount > 200:
                    reward += 1.0
            else:
                # Failed to resolve complaint
                base_penalty = -2.0
                # Penalty scales with order value (opportunity cost)
                if has_order:
                    amount_penalty = -0.5 * math.log10(max(10, order_amount))
                else:
                    amount_penalty = 0
                reward = base_penalty + amount_penalty

        else:
            # Unknown scenario - neutral
            reward = 0.0 if won else -0.5

        logger.info(
            f"[CustomerServiceEnv] Episode reward: {reward:.2f} "
            f"(scenario={scenario}, won={won}, has_order={has_order}, amount={order_amount})"
        )

        return reward

    def get_available_actions(self) -> List[str]:
        """Get list of available actions at current node."""
        if not self.state or not self.current_playbook:
            return []

        nodes = self.current_playbook['nodes']
        current_node_data = nodes.get(self.state.current_node, {})

        # Return available_skills from node, or all valid skills as fallback
        return current_node_data.get('available_skills', list(VALID_SKILLS))

    def render(self) -> str:
        """Render current state as string."""
        if not self.state:
            return "Environment not initialized"

        nodes = self.current_playbook['nodes']
        node = nodes.get(self.state.current_node, {})

        lines = [
            f"=== CustomerServiceEnv ===",
            f"Playbook: {self.current_playbook['playbook_id']}",
            f"Scenario: {self.state.scenario}",
            f"Current Node: {self.state.current_node}",
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
            else:
                lines.append(f"[客服] {turn.get('content', turn.get('action', 'N/A'))}")

        return "\n".join(lines)


# Convenience functions for testing
def create_env(playbook_path: str = "outputs/playbooks.json") -> CustomerServiceEnv:
    """Create and return a CustomerServiceEnv instance."""
    return CustomerServiceEnv(playbook_path)


def run_random_episode(env: CustomerServiceEnv) -> Dict[str, Any]:
    """
    Run a random episode for testing.

    Returns:
        Dict with episode statistics
    """
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    while not env.state.done and steps < 20:
        # Random action from available
        available = env.get_available_actions()
        action = random.choice(available) if available else 'gen_clarify'

        obs, reward, done, step_info = env.step(action)
        total_reward += reward
        steps += 1

    # Add episode reward
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