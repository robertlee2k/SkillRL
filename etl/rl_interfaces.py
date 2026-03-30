# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RL Interfaces for Customer Service Environment.

This module provides the interface layer between:
- CustomerServiceEnv (etl/customer_service_env.py)
- RL Training Framework (agent_system)

It follows the patterns established in agent_system.environments.base and
agent_system.environments.env_manager.
"""

from typing import List, Tuple, Dict, Any, Optional
import re
import logging
import numpy as np
from numpy.typing import NDArray

# Import base class from agent_system
from agent_system.environments.base import EnvironmentManagerBase, to_numpy

# Import customer service environment and skill definitions
from etl.customer_service_env import VALID_SKILLS, HIGH_RISK_SKILLS
from etl.config import SKILL_DEFINITIONS

# Import shared prompt components (DRY principle)
from etl.prompt_config import (
    SKILL_DICT,
    PRIORITY_WATERFALL_RULES,
    format_available_skills_rich,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

INVALID_ACTION = "INVALID_ACTION"
FALLBACK_ACTION_PRIORITY = ["gen_clarify", "gen_empathize", "gen_transfer"]

# Skill ID to index mapping (for discrete action space)
SKILL_ID_TO_IDX: Dict[str, int] = {skill: i for i, skill in enumerate(sorted(VALID_SKILLS))}
IDX_TO_SKILL_ID: Dict[int, str] = {i: skill for skill, i in SKILL_ID_TO_IDX.items()}


# =============================================================================
# Prompt Templates
# =============================================================================

CUSTOMER_SERVICE_TEMPLATE_NO_HIS = """你是一名专业的电商客服智能助手，擅长处理售前咨询、物流查询和售后问题。

{priority_waterfall_rules}

## 当前对话状态

**场景类型**: {scenario}
**买家情绪**: {sentiment}

**买家最新消息**:
{buyer_text}

**当前槽位状态**:
{slots_formatted}

## 可用动作列表

以下是你当前可以选择的动作：

{available_skills_formatted}

## 动作说明

{skill_descriptions}

## 输出格式要求

你**必须**严格按照以下 XML 格式输出，先写思考过程，最后输出动作 ID：

<think>
在这里写下你的分析过程。
</think>
<action>这里填写对应的动作ID</action>

【标准输出示例】：
<think>
买家正在询问这款奶瓶是否带有把手，属于售前关于产品规格的咨询。此时应该选择回答规格问题的动作。
</think>
<action>pre_answer_spec</action>
"""

CUSTOMER_SERVICE_TEMPLATE = """你是一名专业的电商客服智能助手，擅长处理售前咨询、物流查询和售后问题。你的任务是: {task_description}

{priority_waterfall_rules}

## 历史对话记录

在当前步骤之前，你已经执行了 {step_count} 步操作。以下是最近的 {history_length} 条对话和操作记录：

{action_history}

## 当前对话状态

**场景类型**: {scenario}
**当前步骤**: {current_step}
**买家情绪**: {sentiment}

**买家最新消息**:
{buyer_text}

**当前槽位状态**:
{slots_formatted}

## 可用动作列表

以下是你当前可以选择的动作：

{available_skills_formatted}

## 动作说明

{skill_descriptions}

## 输出格式要求

你**必须**严格按照以下 XML 格式输出，先写思考过程，最后输出动作 ID：

<think>
在这里写下你的分析过程。
</think>
<action>这里填写对应的动作ID</action>

【标准输出示例】：
<think>
买家正在询问这款奶瓶是否带有把手，属于售前关于产品规格的咨询。此时应该选择回答规格问题的动作。
</think>
<action>pre_answer_spec</action>
"""


# =============================================================================
# Action Projection (Parsing + Validation)
# =============================================================================

def customer_service_projection(
    actions: List[str],
    available_skills_list: Optional[List[List[str]]] = None
) -> Tuple[List[str], List[int]]:
    """
    Project LLM text outputs to skill IDs with validity checking.

    This function extracts skill_id from LLM output in format:
        <think>analysis</think>
        <action>skill_id</action>

    Validation rules (valids[i] = 0 if any fails):
        1. Extracted skill_id must be in VALID_SKILLS (31 whitelist)
        2. Extracted skill_id must NOT contain parameters (e.g., [order=123])
        3. Extracted skill_id should be in available_skills (if provided)
        4. Extraction must succeed (non-empty result)

    Args:
        actions: List of raw LLM outputs
        available_skills_list: Optional list of available skills per environment.
                               If None, validation only checks against VALID_SKILLS.

    Returns:
        Tuple of:
            - results: List of extracted/interpolated skill IDs
            - valids: List of validity flags (1=valid, 0=invalid)
    """
    results: List[str] = []
    valids: List[int] = [1] * len(actions)

    # Pre-compiled regex patterns
    re_action_block = re.compile(
        r"<action>\s*([a-zA-Z0-9_\[\]=,\s]+?)\s*</action>",
        re.IGNORECASE | re.DOTALL
    )
    re_skill_with_params = re.compile(r"([a-z_]+)\s*\[.*?\]", re.IGNORECASE)
    re_skill_direct = re.compile(
        r"\b(gen_[a-z_]+|pre_[a-z_]+|log_[a-z_]+|aft_[a-z_]+)\b",
        re.IGNORECASE
    )

    for i, action in enumerate(actions):
        original_action = action
        extracted_skill: Optional[str] = None

        # Format validation: enforce thinking process before action
        # This prevents reward hacking where model skips thinking to maximize reward
        think_start_tag = "<think>"
        think_end_tag = "</think>"
        think_start = think_start_tag in action
        think_end = think_end_tag in action

        if not (think_start and think_end):
            results.append(INVALID_ACTION)
            valids[i] = 0
            logger.warning(f"[projection] Format breached (missing think tags): {original_action[:80]}...")
            continue

        # Ensure action tag comes AFTER thinking process ends
        think_end_idx = action.find(think_end_tag)
        action_start_idx = action.find("<action>")
        if action_start_idx != -1 and action_start_idx < think_end_idx:
            results.append(INVALID_ACTION)
            valids[i] = 0
            logger.warning(f"[projection] Format breached (action before think end): {original_action[:80]}...")
            continue

        # Step 1: Try to extract from <action>...</action> block
        match = re_action_block.search(action)
        if match:
            raw_skill = match.group(1).strip()

            # Check if it has parameters (e.g., aft_initiate_refund[order=123])
            param_match = re_skill_with_params.match(raw_skill)
            if param_match:
                # Extract skill without parameters
                extracted_skill = param_match.group(1).lower()
                logger.debug(
                    f"[projection] Found skill with params: {raw_skill} -> {extracted_skill}"
                )
            else:
                extracted_skill = raw_skill.lower()

        # Step 2: Fallback to direct skill_id pattern matching
        if not extracted_skill:
            direct_match = re_skill_direct.search(action)
            if direct_match:
                extracted_skill = direct_match.group(1).lower()
                logger.debug(
                    f"[projection] Found skill via direct match: {extracted_skill}"
                )

        # Step 3: Validate and set result
        if extracted_skill is None:
            # Extraction failed
            results.append(INVALID_ACTION)
            valids[i] = 0
            logger.warning(f"[projection] Failed to extract action from: {original_action[:100]}...")
            continue

        # Validate against global whitelist
        if extracted_skill not in VALID_SKILLS:
            results.append(INVALID_ACTION)
            valids[i] = 0
            logger.warning(
                f"[projection] Invalid skill '{extracted_skill}' not in VALID_SKILLS"
            )
            continue

        # Validate against available_skills if provided
        if available_skills_list is not None and i < len(available_skills_list):
            available_skills = available_skills_list[i]
            if extracted_skill not in available_skills:
                # Skill is valid globally but not available at current node
                # We still pass it through but mark as invalid for metrics
                results.append(extracted_skill)
                valids[i] = 0
                logger.debug(
                    f"[projection] Skill '{extracted_skill}' not in available_skills: {available_skills}"
                )
                continue

        results.append(extracted_skill)

    return results, valids


def customer_service_fallback_projection(
    actions: List[str],
    available_skills_list: List[List[str]]
) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
    """
    Project LLM outputs with fallback interpolation.

    When extraction fails or skill is invalid, this function returns a
    fallback action based on priority: gen_clarify -> gen_empathize -> gen_transfer.

    Args:
        actions: List of raw LLM outputs
        available_skills_list: List of available skills per environment

    Returns:
        Tuple of:
            - results: List of skill IDs (never INVALID_ACTION)
            - valids: List of validity flags (1=valid, 0=fallback used)
            - intercept_infos: List of interception metadata
    """
    results: List[str] = []
    valids: List[int] = [1] * len(actions)
    intercept_infos: List[Dict[str, Any]] = []

    # First pass: standard projection
    extracted_actions, extraction_valids = customer_service_projection(
        actions, available_skills_list
    )

    for i, (action, extracted, is_valid, available) in enumerate(
        zip(actions, extracted_actions, extraction_valids, available_skills_list)
    ):
        intercept_info: Dict[str, Any] = {
            "original_output": action,
            "extracted_action": extracted,
            "was_intercepted": False,
            "intercept_reason": None,
            "fallback_used": None,
        }

        if is_valid == 1 and extracted != INVALID_ACTION:
            # Valid extraction
            results.append(extracted)
        else:
            # Need fallback
            valids[i] = 0
            intercept_info["was_intercepted"] = True

            # Determine fallback
            fallback = _select_fallback_action(available)
            results.append(fallback)

            # Record interception reason
            if extracted == INVALID_ACTION:
                intercept_info["intercept_reason"] = "extraction_failed"
            elif extracted not in VALID_SKILLS:
                intercept_info["intercept_reason"] = "invalid_skill_id"
            elif extracted not in available:
                intercept_info["intercept_reason"] = "skill_not_available"
            else:
                intercept_info["intercept_reason"] = "unknown"

            intercept_info["fallback_used"] = fallback

            logger.info(
                f"[projection] Fallback at idx {i}: '{extracted}' -> '{fallback}' "
                f"(reason: {intercept_info['intercept_reason']})"
            )

        intercept_infos.append(intercept_info)

    return results, valids, intercept_infos


def _select_fallback_action(available_skills: List[str]) -> str:
    """
    Select a fallback action from available skills.

    Priority: gen_clarify -> gen_empathize -> gen_transfer -> first available

    Args:
        available_skills: List of available skill IDs

    Returns:
        A valid skill ID from available_skills
    """
    for priority_skill in FALLBACK_ACTION_PRIORITY:
        if priority_skill in available_skills:
            return priority_skill

    # Fallback to first available
    if available_skills:
        return available_skills[0]

    # Ultimate fallback (should never reach here in practice)
    logger.error("[projection] No available skills! Using gen_clarify as ultimate fallback")
    return "gen_clarify"


# =============================================================================
# Prompt Builder
# =============================================================================

class CustomerServicePromptBuilder:
    """Build prompts for customer service environment."""

    @staticmethod
    def build(
        observation: Dict[str, Any],
        action_history: Optional[List[str]] = None,
        history_length: int = 5
    ) -> str:
        """
        Build a prompt from environment observation.

        Args:
            observation: Observation dict from CustomerServiceEnv
            action_history: List of previous actions taken (deprecated, use dialogue_history)
            history_length: Number of recent dialogue turns to include

        Returns:
            Formatted prompt string
        """
        scenario = observation.get("scenario", "unknown")
        buyer_text = observation.get("buyer_text", "")
        sentiment = observation.get("sentiment", "neutral")
        available_skills = observation.get("available_skills", list(VALID_SKILLS))
        slots = observation.get("slots", {})
        dialogue_history = observation.get("dialogue_history", [])
        action_history = observation.get("action_history", [])

        # Format available skills with rich information (including common_mistakes)
        # NOTE: node_id is NOT exposed to prevent cheating - only skill_name is shown
        available_skills_formatted = format_available_skills_rich(available_skills)

        # Format slots
        if slots:
            slots_formatted = "\n".join([
                f"- {k}: {v}" for k, v in slots.items()
            ])
        else:
            slots_formatted = "（暂无槽位信息）"

        # Format skill descriptions for context
        skill_descriptions = "\n".join([
            f"- `{skill}`: {SKILL_DEFINITIONS.get(skill, {}).get('description', '未知')}"
            for skill in available_skills[:10]  # Limit to avoid prompt bloat
        ])

        # Format dialogue history (full conversation for context)
        if dialogue_history and len(dialogue_history) > 0:
            # Take last N turns
            recent_dialogue = dialogue_history[-(history_length * 2):]
            dialogue_lines = []
            for turn in recent_dialogue:
                if turn['role'] == 'buyer':
                    dialogue_lines.append(f"买家: {turn['content']}")
                elif turn['role'] == 'system':
                    # 【修复1】正确剥离系统警告，不污染客服角色
                    dialogue_lines.append(f"【系统警告】: {turn['content']}")
                else:
                    # 【修复2】优雅处理非法动作的展示
                    action_str = turn.get('action', 'N/A')
                    if action_str == 'INVALID_ACTION':
                        dialogue_lines.append(f"客服: [尝试了无效的动作]")
                    else:
                        dialogue_lines.append(f"客服: [Action: {action_str}]")
            dialogue_text = "\n".join(dialogue_lines)

            action_len = len(action_history) if action_history else 0
            return CUSTOMER_SERVICE_TEMPLATE.format(
                task_description=f"处理{scenario}场景下的客户问题",
                priority_waterfall_rules=PRIORITY_WATERFALL_RULES,
                step_count=action_len,
                history_length=min(history_length, len(dialogue_history) // 2),
                action_history=dialogue_text,
                scenario=scenario,
                current_step=action_len + 1,
                sentiment=sentiment,
                buyer_text=buyer_text,
                slots_formatted=slots_formatted,
                available_skills_formatted=available_skills_formatted,
                skill_descriptions=skill_descriptions,
            )
        else:
            return CUSTOMER_SERVICE_TEMPLATE_NO_HIS.format(
                priority_waterfall_rules=PRIORITY_WATERFALL_RULES,
                scenario=scenario,
                sentiment=sentiment,
                buyer_text=buyer_text,
                slots_formatted=slots_formatted,
                available_skills_formatted=available_skills_formatted,
                skill_descriptions=skill_descriptions,
            )


# =============================================================================
# Environment Manager
# =============================================================================

class CustomerServiceEnvironmentManager(EnvironmentManagerBase):
    """
    Environment manager for CustomerServiceEnv.

    This class wraps CustomerServiceEnv to provide the interface expected
    by the RL training framework (agent_system).

    Key responsibilities:
    1. Manage vectorized environments
    2. Build text observations (prompts) for LLM
    3. Project LLM outputs to environment actions
    4. Track action history and episode statistics
    """

    def __init__(
        self,
        envs: Any,  # Vectorized CustomerServiceEnv
        projection_f: Any,
        config: Any
    ):
        """
        Initialize the environment manager.

        Args:
            envs: Vectorized environment instance
            projection_f: Projection function (customer_service_projection or variant)
            config: Configuration object (OmegaConf or similar)
        """
        # Internal state tracking
        self.action_histories: List[List[str]] = []
        self.step_counts: List[int] = []
        self.tasks: List[str] = []  # Scenario descriptions

        super().__init__(envs, projection_f, config)

    def reset(self, kwargs: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Reset all environments and return initial observations.

        Args:
            kwargs: Optional reset parameters (e.g., specific playbook_ids)

        Returns:
            Tuple of:
                - observations: Dict with 'text', 'image', 'anchor' keys
                - infos: List of info dicts from environment reset
        """
        kwargs = kwargs or {}

        # Initialize tracking state
        self._last_infos: List[Dict[str, Any]] = []
        obs_list: List[Dict[str, Any]] = []
        infos: List[Dict[str, Any]] = []

        # CRITICAL: Must explicitly iterate and reset each underlying environment
        # This ensures CustomerServiceEnv.state is properly initialized
        if hasattr(self.envs, 'envs'):
            # SimpleVectorEnv wrapper with .envs attribute
            for env in self.envs.envs:
                obs, info = env.reset()
                obs_list.append(obs)
                infos.append(info)
        elif hasattr(self.envs, 'reset'):
            # Single environment or other vectorized env
            base_obs, base_infos = self.envs.reset(**kwargs)

            # Normalize to list format
            if isinstance(base_obs, list):
                obs_list = base_obs
            else:
                obs_list = [base_obs]

            if isinstance(base_infos, list):
                infos = base_infos
            else:
                infos = [base_infos]
        else:
            raise RuntimeError("Environment does not have reset method")

        # Store infos for external access
        self._last_infos = infos

        # Initialize tracking state
        batch_size = len(obs_list)
        self.action_histories = [[] for _ in range(batch_size)]
        self.step_counts = [0] * batch_size
        self.tasks = [obs.get("scenario", "unknown") for obs in obs_list]

        # Build text observations
        text_obs = self.build_text_obs(obs_list, init=True)

        observations = {
            "text": text_obs,
            "image": None,  # CustomerServiceEnv is text-only
            "anchor": [obs.copy() for obs in obs_list],  # Store raw observations
        }

        return observations, infos

    def step(
        self,
        text_actions: List[str]
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[Dict]]:
        """
        Execute text actions and return next state.

        Args:
            text_actions: List of LLM outputs (raw text with <action> tags)

        Returns:
            Tuple of:
                - next_observations: Dict with 'text', 'image', 'anchor' keys
                - rewards: np.ndarray of step rewards
                - dones: np.ndarray of done flags
                - infos: List of info dicts with 'is_action_valid' added
        """
        # Get available skills for each environment
        available_skills_list = self._get_available_skills_list()

        # Project text actions to skill IDs
        if self.config.env.get("use_fallback_projection", True):
            actions, valids, intercept_infos = customer_service_fallback_projection(
                text_actions, available_skills_list
            )
        else:
            actions, valids = customer_service_projection(
                text_actions, available_skills_list
            )
            intercept_infos = [{}] * len(actions)

        # Execute actions in environments
        obs_list: List[Dict[str, Any]] = []
        reward_list: List[float] = []
        done_list: List[bool] = []
        infos: List[Dict[str, Any]] = []

        for i, action in enumerate(actions):
            # Execute step
            obs, reward, done, info = self._execute_single_step(i, action)
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)

            # Add validity and interception info
            info["is_action_valid"] = to_numpy(valids[i])
            info["intercept_info"] = intercept_infos[i]
            info["original_text_action"] = text_actions[i]
            infos.append(info)

            # Update tracking
            if not done:
                self.action_histories[i].append(action)
                self.step_counts[i] += 1

        # Build text observations
        text_obs = self.build_text_obs(obs_list)

        next_observations = {
            "text": text_obs,
            "image": None,
            "anchor": [obs.copy() for obs in obs_list],
        }

        rewards: np.ndarray = to_numpy(reward_list)
        dones: np.ndarray = to_numpy(done_list)

        return next_observations, rewards, dones, infos

    def _execute_single_step(
        self,
        env_idx: int,
        action: str
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute a single step in one environment.

        Args:
            env_idx: Index of the environment
            action: Skill ID to execute

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Handle different environment types
        if hasattr(self.envs, 'envs'):
            # Vectorized environment with .envs attribute
            env = self.envs.envs[env_idx]
            return env.step(action)
        elif hasattr(self.envs, 'step'):
            # Try batch step
            results = self.envs.step([action])
            if isinstance(results, tuple) and len(results) == 4:
                obs, reward, done, info = results
                if isinstance(obs, list):
                    obs = obs[0]
                if isinstance(reward, list):
                    reward = reward[0]
                if isinstance(done, list):
                    done = done[0]
                if isinstance(info, list):
                    info = info[0]
                return obs, reward, done, info

        raise RuntimeError("Cannot execute step on environment")

    def _get_available_skills_list(self) -> List[List[str]]:
        """Get available skills for each environment."""
        available_list: List[List[str]] = []

        for i in range(len(self.action_histories)):
            try:
                if hasattr(self.envs, 'envs'):
                    env = self.envs.envs[i]
                    available = env.get_available_actions()
                elif hasattr(self.envs, 'get_available_actions'):
                    available = self.envs.get_available_actions()
                else:
                    available = list(VALID_SKILLS)

                available_list.append(available)
            except Exception as e:
                logger.warning(f"Failed to get available skills for env {i}: {e}")
                available_list.append(list(VALID_SKILLS))

        return available_list

    def build_text_obs(
        self,
        observations: Optional[List[Dict[str, Any]]] = None,
        init: bool = False
    ) -> List[str]:
        """
        Build text observations (prompts) from environment observations.

        This method overrides the base class to provide custom prompt building.
        The base class signature is build_text_obs(self), but subclasses like
        AlfWorldEnvironmentManager use additional parameters.

        Args:
            observations: List of observation dicts (optional for base compatibility)
            init: Whether this is the initial observation (no history)

        Returns:
            List of formatted prompt strings
        """
        # Use stored observations if not provided (base class compatibility)
        if observations is None:
            observations = getattr(self, '_last_observations', [])

        if not observations:
            return []

        postprocess_text_obs: List[str] = []

        for i, obs in enumerate(observations):
            action_history = self.action_histories[i] if i < len(self.action_histories) else []
            # Safely get history_length with proper default handling
            history_length = self.config.env.get("history_length", 5) if hasattr(self.config.env, 'get') else getattr(self.config.env, 'history_length', 5)
            if history_length is None:
                history_length = 5

            prompt = CustomerServicePromptBuilder.build(
                observation=obs,
                action_history=action_history if not init else None,
                history_length=history_length
            )
            postprocess_text_obs.append(prompt)

        return postprocess_text_obs

    def compute_episode_rewards(self) -> List[float]:
        """
        Compute final episode rewards for completed episodes.

        Should be called after episode termination to get business outcome rewards.

        Returns:
            List of episode rewards
        """
        episode_rewards: List[float] = []

        for i in range(len(self.action_histories)):
            try:
                if hasattr(self.envs, 'envs'):
                    env = self.envs.envs[i]
                    reward = env.compute_episode_reward()
                else:
                    reward = 0.0
                episode_rewards.append(reward)
            except Exception as e:
                logger.warning(f"Failed to compute episode reward for env {i}: {e}")
                episode_rewards.append(0.0)

        return episode_rewards

    def _process_batch(
        self,
        batch_idx: int,
        total_batch_list: List[Any],
        total_infos: List[List[Dict]],
        success: Dict[str, List]
    ) -> None:
        """
        Process a batch for success evaluation.

        Called by success_evaluator to extract success metrics.
        """
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item.get('active_masks', False):
                info = total_infos[batch_idx][i]

                # Extract success metrics
                won_value = float(info.get('won', 0))
                success['success_rate'].append(won_value)

                # Extract valid action rate for TensorBoard monitoring
                success['valid_action_rate'].append(float(info.get('is_action_valid', 0)))

                # Extract scenario-specific metrics
                scenario = info.get('scenario', 'unknown')
                success[f'{scenario}_success_rate'].append(won_value)

                # Extract business metrics
                business_outcome = info.get('business_outcome', {})
                if business_outcome.get('has_order', False):
                    success['order_rate'].append(1.0)
                    success['order_amount'].append(business_outcome.get('order_amount', 0.0))
                else:
                    success['order_rate'].append(0.0)

                # Extract success rate by turn length (分段成功率)
                for key in info:
                    if key.startswith('success_rate/'):
                        success[key].append(float(info[key]))

                return


# =============================================================================
# Factory Functions
# =============================================================================

def build_customer_service_envs(
    playbook_path: str,
    env_num: int,
    seed: int = 42,
    **kwargs
) -> Any:
    """
    Build vectorized customer service environments.

    Args:
        playbook_path: Path to playbooks JSON file
        env_num: Number of parallel environments
        seed: Random seed
        **kwargs: Additional arguments

    Returns:
        Vectorized environment instance
    """
    from etl.customer_service_env import CustomerServiceEnv

    if env_num == 1:
        return CustomerServiceEnv(playbook_path)

    # For multiple environments, create a simple wrapper
    # In production, would use proper vectorization (e.g., gym.vector.SyncVectorEnv)
    class SimpleVectorEnv:
        """Simple synchronous vectorized environment."""

        def __init__(self, playbook_path: str, num_envs: int):
            self.envs = [
                CustomerServiceEnv(playbook_path)
                for _ in range(num_envs)
            ]
            self.num_envs = num_envs

        def reset(self, **kwargs) -> Tuple[List[Dict], List[Dict]]:
            obs_list = []
            info_list = []
            for env in self.envs:
                obs, info = env.reset()
                obs_list.append(obs)
                info_list.append(info)
            return obs_list, info_list

        def step(self, actions: List[str]) -> Tuple[List[Dict], List[float], List[bool], List[Dict]]:
            obs_list = []
            rewards = []
            dones = []
            infos = []

            for i, (env, action) in enumerate(zip(self.envs, actions)):
                obs, reward, done, info = env.step(action)
                obs_list.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)

            return obs_list, rewards, dones, infos

        def close(self) -> None:
            """Close all environments."""
            for env in self.envs:
                # CustomerServiceEnv doesn't have close method, use getattr for safety
                close_method = getattr(env, 'close', None)  # type: ignore[union-attr]
                if close_method is not None:
                    close_method()

    return SimpleVectorEnv(playbook_path, env_num)


def make_customer_service_envs(
    config: Any
) -> Tuple[CustomerServiceEnvironmentManager, CustomerServiceEnvironmentManager]:
    """
    Create training and validation environment managers.

    This follows the pattern in env_manager.make_envs().

    Args:
        config: Configuration object with:
            - config.env.playbook_path: Path to playbooks
            - config.env.seed: Random seed
            - config.data.train_batch_size: Number of training envs
            - config.data.val_batch_size: Number of validation envs

    Returns:
        Tuple of (train_envs, val_envs)
    """
    from functools import partial

    playbook_path = config.env.playbook_path
    seed = config.env.get("seed", 42)
    train_batch_size = config.data.train_batch_size
    val_batch_size = config.data.val_batch_size

    # Build base environments
    _train_envs = build_customer_service_envs(
        playbook_path=playbook_path,
        env_num=train_batch_size,
        seed=seed
    )
    _val_envs = build_customer_service_envs(
        playbook_path=playbook_path,
        env_num=val_batch_size,
        seed=seed + 1000
    )

    # Create projection function
    projection_f = partial(customer_service_fallback_projection)

    # Create environment managers
    train_envs = CustomerServiceEnvironmentManager(_train_envs, projection_f, config)
    val_envs = CustomerServiceEnvironmentManager(_val_envs, projection_f, config)

    logger.info(
        f"[CustomerServiceEnvironmentManager] Created {train_batch_size} train envs, "
        f"{val_batch_size} val envs"
    )

    return train_envs, val_envs


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "VALID_SKILLS",
    "HIGH_RISK_SKILLS",
    "INVALID_ACTION",
    "SKILL_ID_TO_IDX",
    "IDX_TO_SKILL_ID",
    # Projection functions
    "customer_service_projection",
    "customer_service_fallback_projection",
    # Prompt builder
    "CustomerServicePromptBuilder",
    # Environment manager
    "CustomerServiceEnvironmentManager",
    # Factory functions
    "build_customer_service_envs",
    "make_customer_service_envs",
]
