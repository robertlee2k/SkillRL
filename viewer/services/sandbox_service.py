"""Sandbox Service - 沙盒推演服务

提供模型加载、会话管理、异步推理等功能。
"""

import re
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class SandboxSession:
    """单个沙盒会话状态"""
    session_id: str
    playbook_id: str
    env: Any  # CustomerServiceEnv
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    total_reward: float = 0.0
    step_count: int = 0
    done: bool = False
    won: bool = False


class ModelManager:
    """
    单例模型加载管理器。

    模型在首次使用时懒加载，之后保持在内存中。
    """
    _instance: Optional['ModelManager'] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model = None
        self.tokenizer = None
        self.checkpoint_path = None
        self._initialized = True

    def load_model(self, checkpoint_path: str):
        """
        加载模型。如果已经加载且路径相同，则跳过。
        """
        if self.model is not None and self.checkpoint_path == checkpoint_path:
            logger.info(f"Model already loaded from {checkpoint_path}")
            return

        logger.info(f"Loading model from {checkpoint_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            fix_mistral_regex=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        self.checkpoint_path = checkpoint_path

        logger.info(f"Model loaded on {self.model.device}")

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None and self.tokenizer is not None

    def unload_model(self):
        """
        卸载模型，释放 GPU 显存。
        """
        if self.model is None:
            logger.info("No model to unload")
            return

        logger.info(f"Unloading model from {self.checkpoint_path}")

        # 删除模型和tokenizer引用
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self.checkpoint_path = None

        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU cache cleared")

        logger.info("Model unloaded successfully")


# 全局单例
model_manager = ModelManager()


class SessionManager:
    """
    内存态会话管理器，支持自动过期清理。
    """

    def __init__(self, max_age_minutes: int = 30):
        self.sessions: Dict[str, SandboxSession] = {}
        self.max_age_minutes = max_age_minutes

    def create(self, playbook_id: str, env: Any) -> SandboxSession:
        """
        创建新会话。

        Args:
            playbook_id: 剧本 ID
            env: CustomerServiceEnv 实例

        Returns:
            新创建的 SandboxSession
        """
        session_id = str(uuid.uuid4())[:8]
        session = SandboxSession(
            session_id=session_id,
            playbook_id=playbook_id,
            env=env
        )
        self.sessions[session_id] = session
        logger.info(f"Created session {session_id} for playbook {playbook_id}")
        return session

    def get(self, session_id: str) -> Optional[SandboxSession]:
        """
        获取会话。如果不存在或已过期，返回 None。
        """
        session = self.sessions.get(session_id)
        if session is None:
            return None

        # 检查是否过期
        age = datetime.now() - session.last_active
        if age > timedelta(minutes=self.max_age_minutes):
            logger.info(f"Session {session_id} expired, removing")
            del self.sessions[session_id]
            return None

        return session

    def update_activity(self, session_id: str):
        """更新会话活动时间"""
        session = self.sessions.get(session_id)
        if session:
            session.last_active = datetime.now()

    def remove(self, session_id: str):
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Removed session {session_id}")

    def cleanup_expired(self):
        """清理所有过期会话"""
        now = datetime.now()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session.last_active > timedelta(minutes=self.max_age_minutes)
        ]
        for sid in expired:
            del self.sessions[sid]
            logger.info(f"Cleaned up expired session {sid}")

        return len(expired)


# 全局会话管理器
session_manager = SessionManager(max_age_minutes=30)


def parse_model_output(output: str) -> Tuple[str, Optional[str]]:
    """
    解析模型输出，提取动作和推理过程。

    使用与训练一致的解析逻辑 (etl.rl_interfaces.customer_service_projection)。

    Args:
        output: 模型生成的原始输出

    Returns:
        (action, reasoning) 元组，reasoning 可能为 None
    """
    # 使用训练时的解析逻辑
    from etl.rl_interfaces import customer_service_projection

    # customer_service_projection 返回 (actions, valids)
    actions, valids = customer_service_projection([output], available_skills_list=None)

    action = actions[0] if actions else ""

    # 提取推理过程 - 使用训练时的格式 <think>...</think>
    reasoning_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None

    return action, reasoning


def _sync_generate(model, tokenizer, prompt: str, temperature: float = 0.4, max_new_tokens: int = 512) -> str:
    """
    同步生成函数，用于在 executor 中运行。
    """
    logger.info(f"[generate] Prompt length: {len(prompt)} chars")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    logger.info(f"[generate] Input tokens: {inputs['input_ids'].shape}")

    # 确保 pad_token_id 和 eos_token_id 有效
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id

    logger.info(f"[generate] pad_token_id: {pad_token_id}, eos_token_id: {eos_token_id}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

    logger.info(f"[generate] Output tokens shape: {outputs.shape}")

    # 只解码新生成的 token
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    logger.info(f"[generate] New tokens count: {len(new_tokens)}")

    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    logger.info(f"[generate] Response length: {len(response)} chars, preview: {response[:200]}...")

    # 去掉开头的空白
    response = response.lstrip()
    logger.info(f"[generate] After lstrip, length: {len(response)} chars")

    # 截断到第一个 </action> 后（模型可能输出多个action，只取第一个）
    if '</action>' in response:
        # 找到第一个 </action> 的位置，截取到它之后
        end_pos = response.find('</action>') + len('</action>')
        response = response[:end_pos].rstrip()
        logger.info(f"[generate] Truncated at first </action>, length: {len(response)} chars")

    # 截断到第一个 Human: 或买家: 出现的位置（防止模型继续生成多轮对话）
    for stop in ["Human:", "买家:"]:
        if stop in response:
            response = response.split(stop)[0].rstrip()

    return response


async def generate_action_async(
    model,
    tokenizer,
    prompt: str,
    temperature: float = 0.4,
    max_new_tokens: int = 512
) -> str:
    """
    异步生成模型回复。

    使用 asyncio.to_thread 将阻塞的 model.generate 调用放到线程池中，
    避免阻塞 FastAPI 的事件循环。

    Args:
        model: Transformers 模型
        tokenizer: 分词器
        prompt: 输入提示词
        temperature: 采样温度
        max_new_tokens: 最大新生成 token 数

    Returns:
        模型生成的文本
    """
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,  # 使用默认线程池
        lambda: _sync_generate(model, tokenizer, prompt, temperature, max_new_tokens)
    )
    return response


def build_prompt_from_observation(
    observation: Dict[str, Any],
    action_history: Optional[List[str]] = None,
    history_length: int = 5
) -> str:
    """
    从环境观察构建模型输入提示词。

    使用与训练相同的 CustomerServicePromptBuilder。

    Args:
        observation: 环境观察（已包含 dialogue_history）
        action_history: 动作历史
        history_length: 历史长度

    Returns:
        格式化的提示词字符串
    """
    # 导入训练时使用的 PromptBuilder
    import sys
    sys.path.insert(0, '/home/bo.li/SkillRL')
    from etl.rl_interfaces import CustomerServicePromptBuilder

    return CustomerServicePromptBuilder.build(
        observation=observation,
        action_history=action_history,
        history_length=history_length
    )