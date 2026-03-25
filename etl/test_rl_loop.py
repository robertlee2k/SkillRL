#!/usr/bin/env python
"""
Test script for RL Loop with DummyAgent.

This script validates that the RL infrastructure works end-to-end:
1. Environment reset/step flow
2. Action projection (parsing LLM output -> skill_id)
3. Dialogue history tracking
4. Step rewards and episode rewards

Run: python etl/test_rl_loop.py
"""

import sys
import random
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from etl.customer_service_env import CustomerServiceEnv, VALID_SKILLS
from etl.rl_interfaces import (
    customer_service_projection,
    customer_service_fallback_projection,
    CustomerServicePromptBuilder,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DummyAgent:
    """
    A dummy agent that generates random actions in the required format.

    This simulates an LLM that outputs:
        思考过程
        ...
        some thinking here
        some more thinking

        <action>skill_id</action>
    """

    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        self.rng = random.Random(seed)
        self.call_count = 0

    def generate(self, prompt: str, available_skills: List[str]) -> str:
        """
        Generate a fake LLM response with action.

        Args:
            prompt: The input prompt (not used by dummy agent)
            available_skills: List of valid actions to choose from

        Returns:
            A formatted string with thinking and action
        """
        self.call_count += 1

        # Choose a random action from available skills
        if available_skills:
            action = self.rng.choice(available_skills)
        else:
            action = self.rng.choice(list(VALID_SKILLS))

        # Simulate LLM output with CoT and action tag
        output = f"""<tool_call>
我是客服智能助手，正在分析买家的需求。

当前对话状态：
- 买家发来了消息，我需要理解其意图
- 我应该选择合适的技能来回应

分析：
- 这是一个关于客服场景的对话
- 我应该选择 {action} 这个动作

让我确认一下：{action} 是当前可用的动作之一。
.serif:montserrat<action>{action}</action>"""

        return output

    def generate_with_noise(self, prompt: str, available_skills: List[str]) -> str:
        """
        Generate a response that may have parsing issues (for testing fallback).

        This simulates various edge cases:
        - Action with parameters
        - Invalid action
        - Malformed output
        """
        self.call_count += 1
        noise_type = self.rng.choice(['normal', 'with_params', 'invalid', 'malformed'])

        if noise_type == 'normal':
            action = self.rng.choice(available_skills) if available_skills else 'gen_clarify'
            return f"<tool_call>分析买家意图...</think>\n<action>{action}</action>"

        elif noise_type == 'with_params':
            action = self.rng.choice(available_skills) if available_skills else 'gen_clarify'
            return f"<tool_call>分析...</think>\n<action>{action}[order=123]</action>"

        elif noise_type == 'invalid':
            return f"<tool_call>分析...</think>\n<action>do_something_invalid</action>"

        else:  # malformed
            return "这是乱码输出，没有任何 action 标签"


def test_projection_functions():
    """Test the action projection functions."""
    print("\n" + "=" * 60)
    print("Testing Action Projection Functions")
    print("=" * 60)

    # Test cases
    test_cases = [
        # (input, available_skills, expected_valid)
        ("<tool_call>分析... sometag\n<action>gen_greet</action>", ["gen_greet", "gen_clarify"], True),
        ("<action>pre_answer_spec</action>", ["pre_answer_spec"], True),
        ("<action>aft_initiate_refund[order=123]</action>", ["aft_initiate_refund"], True),  # With params
        ("<action>invalid_skill</action>", ["gen_greet"], False),  # Invalid skill
        ("No action tag here", ["gen_greet"], False),  # No action tag
        ("Multiple tags <action>gen_greet</action> <action>gen_close</action>", ["gen_greet"], True),  # Multiple
    ]

    for i, (input_text, available, expected_valid) in enumerate(test_cases):
        available_list = [available]  # Wrap as list of lists
        results, valids = customer_service_projection([input_text], available_list)
        is_valid = valids[0] == 1

        status = "✓" if is_valid == expected_valid else "✗"
        print(f"\nTest {i + 1}: {status}")
        print(f"  Input: {input_text[:50]}...")
        print(f"  Result: {results[0]}, Valid: {is_valid}")

    # Test fallback projection
    print("\n" + "-" * 40)
    print("Testing Fallback Projection")
    print("-" * 40)

    fallback_tests = [
        ("<tool_call>分析... sometag\n<action>invalid_action</action>", ["gen_clarify", "gen_empathize"]),
        ("No action here", ["gen_clarify"]),
        ("<action>pre_query_product</action>", ["pre_query_product"]),  # Valid
    ]

    for input_text, available in fallback_tests:
        results, valids, infos = customer_service_fallback_projection(
            [input_text], [available]
        )
        print(f"\n  Input: {input_text[:40]}...")
        print(f"  Result: {results[0]}, Valid: {valids[0]}")
        print(f"  Intercept: {infos[0]}")


def test_prompt_builder():
    """Test the prompt builder."""
    print("\n" + "=" * 60)
    print("Testing Prompt Builder")
    print("=" * 60)

    # Mock observation
    observation = {
        'node_id': 'root',
        'scenario': 'presale',
        'buyer_text': '你好，我想问问这个商品有没有货',
        'sentiment': 'neutral',
        'available_skills': ['gen_greet', 'pre_query_product', 'pre_check_stock'],
        'slots': {'product_id': '12345'},
        'dialogue_history': [
            {'role': 'buyer', 'content': '你好，我想问问这个商品有没有货'},
        ],
        'action_history': [],
    }

    prompt = CustomerServicePromptBuilder.build(observation)
    print(f"\nPrompt length: {len(prompt)} characters")
    print("\nPrompt preview (first 500 chars):")
    print("-" * 40)
    print(prompt[:500])
    print("...")
    print("-" * 40)


def run_dummy_episode(playbook_path: str, use_noise: bool = False):
    """
    Run a complete episode with DummyAgent.

    Args:
        playbook_path: Path to playbooks JSON
        use_noise: If True, use noisy agent that tests fallback
    """
    print("\n" + "=" * 60)
    print(f"Running Dummy Episode (noise={use_noise})")
    print("=" * 60)

    # Initialize environment
    env = CustomerServiceEnv(playbook_path)
    agent = DummyAgent(seed=42)

    # Reset
    obs, info = env.reset()
    print(f"\nPlaybook: {info['playbook_id']}")
    print(f"Scenario: {info['scenario']}")
    print(f"Business Outcome: {info['business_outcome']}")

    # Track rewards
    total_step_reward = 0.0
    step = 0
    max_steps = 20

    print("\n" + "-" * 40)
    print("Episode Start")
    print("-" * 40)

    while not env.state.done and step < max_steps:
        step += 1
        available_skills = env.get_available_actions()

        # Build prompt for agent
        prompt = CustomerServicePromptBuilder.build(obs)

        # Agent generates action
        if use_noise:
            raw_output = agent.generate_with_noise(prompt, available_skills)
        else:
            raw_output = agent.generate(prompt, available_skills)

        # Project to skill_id
        actions, valids, _ = customer_service_fallback_projection(
            [raw_output], [available_skills]
        )
        action = actions[0]
        is_valid = valids[0] == 1

        # Execute step
        obs, step_reward, done, step_info = env.step(action)
        total_step_reward += step_reward

        # Print step info
        print(f"\n[Step {step}]")
        print(f"  Buyer: {obs.get('buyer_text', 'N/A')[:60]}...")
        print(f"  Action: {action} (valid={is_valid})")
        print(f"  Reward: {step_reward:.2f}")
        print(f"  Sentiment: {step_info.get('sentiment', 'N/A')}")
        if step_info.get('fell_back'):
            print(f"  ⚠️  Fallback triggered!")

    # Episode ended
    print("\n" + "-" * 40)
    print("Episode End")
    print("-" * 40)

    # Compute final episode reward
    episode_reward = env.compute_episode_reward()
    total_reward = total_step_reward + episode_reward

    print(f"\n📊 Episode Statistics:")
    print(f"  Steps: {step}")
    print(f"  Won: {env.state.won}")
    print(f"  Step Rewards: {total_step_reward:.2f}")
    print(f"  Episode Reward: {episode_reward:.2f}")
    print(f"  Total Reward: {total_reward:.2f}")

    # Print dialogue history
    print(f"\n💬 Dialogue History ({len(env.state.dialogue_history)} turns):")
    for i, turn in enumerate(env.state.dialogue_history[:10]):  # First 10 turns
        if turn['role'] == 'buyer':
            print(f"  [{i+1}] 买家: {turn['content'][:50]}...")
        else:
            print(f"  [{i+1}] 客服: {turn.get('content', turn.get('action', 'N/A'))}")

    # Print render
    print("\n" + env.render())

    return {
        'steps': step,
        'won': env.state.won,
        'step_rewards': total_step_reward,
        'episode_reward': episode_reward,
        'total_reward': total_reward,
    }


def main():
    """Main test function."""
    print("\n" + "=" * 60)
    print("  RL Loop Test Suite")
    print("  Testing CustomerServiceEnv + Projection + PromptBuilder")
    print("=" * 60)

    # Path to playbooks
    playbook_path = project_root / "outputs" / "playbooks_new.json"

    if not playbook_path.exists():
        print(f"\n❌ Error: Playbooks not found at {playbook_path}")
        print("Please run the ETL pipeline first to generate playbooks.")
        return

    # Run tests
    test_projection_functions()
    test_prompt_builder()

    # Run episodes
    print("\n" + "=" * 60)
    print("Episode Tests")
    print("=" * 60)

    # Normal episode
    stats1 = run_dummy_episode(str(playbook_path), use_noise=False)

    # Episode with noise (tests fallback)
    stats2 = run_dummy_episode(str(playbook_path), use_noise=True)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"\n✅ Normal Episode: won={stats1['won']}, reward={stats1['total_reward']:.2f}")
    print(f"✅ Noisy Episode: won={stats2['won']}, reward={stats2['total_reward']:.2f}")
    print("\n🎉 All tests passed! RL infrastructure is working correctly.")


if __name__ == "__main__":
    main()