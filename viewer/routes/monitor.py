"""训练监控路由"""
import json
from pathlib import Path
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse

from ..templates_config import templates
from ..data import data_loader
from ..config import PROJECT_ROOT

router = APIRouter()

# 默认训练日志目录
DEFAULT_LOG_DIR = PROJECT_ROOT / "tensorboard_log"


@router.get("/monitor", response_class=HTMLResponse)
async def monitor_page(request: Request, log_dir: str = Query(None)):
    """训练监控页面"""
    log_path = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    return templates.TemplateResponse(
        "monitor.html",
        {
            "request": request,
            "log_dir": str(log_path),
            "default_log_dir": str(DEFAULT_LOG_DIR),
        }
    )


@router.get("/api/monitor/stats")
async def get_stats(log_dir: str = Query(None)):
    """获取统计数据"""
    playbooks = data_loader.playbooks

    # 成单率
    total = len(playbooks)
    ordered = sum(1 for p in playbooks if p.business_outcome.has_order)

    # scenario 分布
    scenario_counts = {}
    for p in playbooks:
        scenario_counts[p.scenario] = scenario_counts.get(p.scenario, 0) + 1

    # 节点数量分布
    node_counts = [len(p.nodes) for p in playbooks]
    avg_nodes = sum(node_counts) / len(node_counts) if node_counts else 0

    # 技能使用频率
    skill_counts = {}
    for p in playbooks:
        for node in p.nodes.values():
            for skill in node.available_skills:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1

    top_skills = sorted(skill_counts.items(), key=lambda x: -x[1])[:10]

    return {
        "total_playbooks": total,
        "order_rate": ordered / total if total > 0 else 0,
        "scenario_distribution": scenario_counts,
        "avg_nodes": round(avg_nodes, 2),
        "top_skills": top_skills,
    }


@router.get("/api/monitor/training-status")
async def get_training_status(log_dir: str = Query(None)):
    """获取训练状态（从TensorBoard日志读取）"""
    log_path = Path(log_dir) if log_dir else DEFAULT_LOG_DIR

    result = {
        "log_dir": str(log_path),
        "log_exists": log_path.exists(),
        "current_epoch": None,
        "current_step": None,
        "recent_metrics": [],
        "recent_playbooks": [],
    }

    # 读取最近的训练指标（如果有TensorBoard日志）
    if log_path.exists():
        try:
            from tensorboard.backend.event_processing import event_file_loader
            event_files = sorted(log_path.glob("events.out.tfevents.*"), reverse=True)
            if event_files:
                loader = event_file_loader.EventFileLoader(str(event_files[0]))
                events = list(loader)
                for event in events[-10:]:  # 最近10个事件
                    if event.HasField("summary"):
                        for value in event.summary.value:
                            result["recent_metrics"].append({
                                "tag": value.tag,
                                "value": value.simple_value if value.HasField("simple_value") else None,
                                "step": event.step,
                            })
                        if event.step:
                            result["current_step"] = event.step
        except ImportError:
            result["error"] = "tensorboard not installed"
        except Exception as e:
            result["error"] = str(e)

    # 返回最近采样的playbook（从最新文件读取，如果有）
    sample_file = log_path / "recent_playbooks.json"
    if sample_file.exists():
        try:
            with open(sample_file) as f:
                result["recent_playbooks"] = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    return result