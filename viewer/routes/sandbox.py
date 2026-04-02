"""Sandbox 沙盒推演路由

提供 RL Agent 动态推演沙盒的 API 接口。
"""

import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..templates_config import templates
from ..config import DEFAULT_CHECKPOINT_PATH, DEFAULT_PLAYBOOK_PATH, PROJECT_ROOT
from ..services.sandbox_service import (
    model_manager,
    session_manager,
    SandboxSession,
    generate_action_async,
    parse_model_output,
    build_prompt_from_observation
)
from ..utils.action_mapper import get_action_nlg

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sandbox", tags=["sandbox"])


# ==================== Request/Response Models ====================

class InitRequest(BaseModel):
    """初始化请求"""
    playbook_id: str
    checkpoint_path: Optional[str] = None


class StepRequest(BaseModel):
    """单步推演请求"""
    session_id: str


class AutoRunRequest(BaseModel):
    """自动运行请求"""
    session_id: str
    max_steps: int = 20
    delay_ms: int = 300


class CheckpointUpdateRequest(BaseModel):
    """更新模型路径请求"""
    checkpoint_path: str


# ==================== Page Routes ====================

@router.get("", response_class=HTMLResponse)
async def sandbox_page(request: Request):
    """沙盒页面"""
    return templates.TemplateResponse(
        "sandbox.html",
        {
            "request": request,
            "default_checkpoint_path": DEFAULT_CHECKPOINT_PATH,
        }
    )


# ==================== API Routes ====================

@router.get("/api/playbooks")
async def list_playbooks():
    """获取可用的剧本列表"""
    from ..data import data_loader

    playbooks = data_loader.playbooks
    result = []

    for p in playbooks[:100]:  # 限制返回前100个
        result.append({
            "playbook_id": p.playbook_id,
            "scenario": p.scenario,
            "rl_steps": p.rl_steps,
            "has_order": p.business_outcome.has_order if p.business_outcome else False
        })

    return {"playbooks": result, "total": len(playbooks)}


@router.get("/api/checkpoint")
async def get_checkpoint():
    """获取当前模型路径"""
    return {
        "checkpoint_path": model_manager.checkpoint_path or DEFAULT_CHECKPOINT_PATH,
        "is_loaded": model_manager.is_loaded()
    }


@router.post("/api/checkpoint")
async def update_checkpoint(request: CheckpointUpdateRequest):
    """更新模型路径"""
    try:
        model_manager.load_model(request.checkpoint_path)
        return {
            "success": True,
            "checkpoint_path": request.checkpoint_path
        }
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.delete("/api/checkpoint")
async def unload_checkpoint():
    """卸载模型，释放 GPU 显存"""
    try:
        model_manager.unload_model()
        return {
            "success": True,
            "message": "Model unloaded, GPU memory released"
        }
    except Exception as e:
        logger.error(f"Failed to unload model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")


@router.post("/api/init")
async def init_sandbox(request: InitRequest):
    """
    初始化沙盒环境。

    - 加载模型（如果未加载）
    - 创建新的 CustomerServiceEnv 实例
    - 重置环境到指定 playbook
    - 返回初始状态和 session_id
    """
    try:
        # 确定模型路径
        checkpoint_path = request.checkpoint_path or DEFAULT_CHECKPOINT_PATH

        # 加载模型（如果未加载）
        if not model_manager.is_loaded():
            logger.info(f"Loading model from {checkpoint_path}")
            model_manager.load_model(checkpoint_path)

        # 创建环境
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from etl.customer_service_env import CustomerServiceEnv

        env = CustomerServiceEnv(DEFAULT_PLAYBOOK_PATH)

        # 重置环境
        observation, info = env.reset(playbook_id=request.playbook_id)

        # 创建会话
        session = session_manager.create(request.playbook_id, env)

        # 构建响应
        return {
            "session_id": session.session_id,
            "playbook_id": request.playbook_id,
            "scenario": info.get("scenario", "unknown"),
            "observation": _format_observation(observation),
            "state": {
                "node_id": observation.get("node_id", "root"),
                "patience": observation.get("patience", 2),
                "done": False,
                "won": False,
                "step_count": 0,
                "total_reward": 0.0,
            },
            "dialogue": _format_dialogue(observation.get("dialogue_history", [])),
            "xray": {
                "thought": None,
                "action": None,
                "available_skills": observation.get("available_skills", []),
            }
        }

    except ValueError as e:
        logger.error(f"Init failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Init failed: {e}")
        raise HTTPException(status_code=500, detail=f"Init failed: {str(e)}")


@router.post("/api/step")
async def step_sandbox(request: StepRequest):
    """
    执行单步推演。

    - 获取模型输出
    - 解析 action 和 thought
    - 执行 env.step(action)
    - 返回更新后的状态
    """
    session = session_manager.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    if session.done:
        return {
            "session_id": request.session_id,
            "error": "Episode already done",
            "state": _get_session_state(session)
        }

    try:
        env = session.env
        observation = env._get_observation()

        # 构建提示词
        prompt = build_prompt_from_observation(
            observation=observation,
            action_history=env.state.action_history if env.state.action_history else None,
            history_length=5
        )

        # 异步生成模型输出
        model_output = await generate_action_async(
            model_manager.model,
            model_manager.tokenizer,
            prompt,
            temperature=0.4,
            max_new_tokens=512
        )
        logger.info(f"[sandbox] model_output length: {len(model_output)}, content: {model_output[:200]}...")

        # 解析输出
        action, thought = parse_model_output(model_output)
        logger.info(f"[sandbox] parsed action: {action}, thought: {thought}")

        # 如果解析失败，使用默认兜底动作
        if not action:
            action = "gen_clarify"
            thought = f"[解析失败] 原始输出: {model_output[:100]}..."

        # 执行环境步骤
        obs, reward, done, step_info = env.step(action)

        # 更新会话状态
        session.total_reward += reward
        session.step_count += 1
        session.done = done
        session.won = step_info.get("won", False)
        session_manager.update_activity(request.session_id)

        # 获取最新的系统消息（如果有）
        system_message = None
        if env.state.dialogue_history:
            last_msg = env.state.dialogue_history[-1]
            if last_msg.get("role") == "system":
                system_message = last_msg.get("content")

        return {
            "session_id": request.session_id,
            "step": session.step_count,
            "raw_output": model_output,
            "xray": {
                "thought": thought,
                "action": action,
                "action_nlg": get_action_nlg(action),
                "reward": reward,
                "patience": step_info.get("patience", env.state.patience),
                "fell_back": step_info.get("fell_back", False),
                "sentiment": step_info.get("sentiment", "neutral"),
                "system_message": system_message,
                "available_skills": obs.get("available_skills", []),
            },
            "state": {
                "node_id": obs.get("node_id", "root"),
                "patience": step_info.get("patience", env.state.patience),
                "done": done,
                "won": step_info.get("won", False),
                "step_count": session.step_count,
                "total_reward": round(session.total_reward, 3),
            },
            "dialogue": _format_dialogue(env.state.dialogue_history),
        }

    except Exception as e:
        logger.error(f"Step failed: {e}")
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@router.post("/api/autorun")
async def autorun_sandbox(request: AutoRunRequest):
    """
    自动运行直到结束。

    - 循环执行 step 直到 done 或达到 max_steps
    - 每步之间有延迟（delay_ms）
    - 返回所有步骤的记录
    """
    session = session_manager.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    if session.done:
        return {
            "session_id": request.session_id,
            "error": "Episode already done",
            "steps": [],
            "final_state": _get_session_state(session)
        }

    steps = []
    max_steps = min(request.max_steps, 50)  # 硬性上限
    delay = request.delay_ms / 1000.0  # 转换为秒

    try:
        while not session.done and len(steps) < max_steps:
            # 复用 step 逻辑
            step_request = StepRequest(session_id=request.session_id)
            step_result = await step_sandbox(step_request)

            steps.append({
                "step": step_result.get("step", len(steps) + 1),
                "xray": step_result.get("xray", {}),
                "state": step_result.get("state", {}),
            })

            # 延迟
            if delay > 0 and not session.done:
                await asyncio.sleep(delay)

        return {
            "session_id": request.session_id,
            "total_steps": len(steps),
            "steps": steps,
            "final_state": _get_session_state(session),
            "dialogue": _format_dialogue(session.env.state.dialogue_history),
        }

    except Exception as e:
        logger.error(f"AutoRun failed: {e}")
        raise HTTPException(status_code=500, detail=f"AutoRun failed: {str(e)}")


@router.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    session_manager.remove(session_id)
    return {"success": True, "message": f"Session {session_id} removed"}


@router.get("/api/sessions")
async def list_sessions():
    """列出所有活跃会话"""
    sessions = []
    for sid, session in session_manager.sessions.items():
        sessions.append({
            "session_id": sid,
            "playbook_id": session.playbook_id,
            "step_count": session.step_count,
            "done": session.done,
            "won": session.won,
            "created_at": session.created_at.isoformat(),
            "last_active": session.last_active.isoformat(),
        })
    return {"sessions": sessions, "count": len(sessions)}


# ==================== Helper Functions ====================

def _format_observation(observation: dict) -> dict:
    """格式化观察数据供前端使用"""
    return {
        "node_id": observation.get("node_id", "root"),
        "buyer_text": observation.get("buyer_text", ""),
        "sentiment": observation.get("sentiment", "neutral"),
        "patience": observation.get("patience", 2),
        "scenario": observation.get("scenario", "unknown"),
    }


def _format_dialogue(dialogue_history: list) -> list:
    """格式化对话历史供前端显示"""
    formatted = []
    for msg in dialogue_history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # 获取 action 并转换为自然语言
        action = msg.get("action")
        action_nlg = get_action_nlg(action) if action else None

        formatted.append({
            "role": role,
            "content": content,
            "action": action,
            "action_nlg": action_nlg,
        })

    return formatted


def _get_session_state(session: SandboxSession) -> dict:
    """获取会话状态摘要"""
    return {
        "session_id": session.session_id,
        "playbook_id": session.playbook_id,
        "step_count": session.step_count,
        "total_reward": round(session.total_reward, 3),
        "done": session.done,
        "won": session.won,
        "patience": session.env.state.patience if session.env.state else 0,
    }