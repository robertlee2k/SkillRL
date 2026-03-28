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
        # 分类指标
        "validation": {},  # 验证集指标
        "episode": {},     # Episode指标
        "training": {},    # 训练进度
        "actor": {},       # Actor loss指标
        "critic": {},      # Critic指标
        "performance": {}, # 性能指标
        "success_by_length": {},  # 分段成功率
        "val_success_by_length": {},  # 验证集分段成功率
        "recent_playbooks": [],
    }

    # 定义重要指标
    key_metrics = {
        "validation": [
            "val/success_rate",
            "val/unknown_success_rate",
            "val/customer_service/test_score",
            "val/customer_service/tool_call_count/mean",
            # 按场景分的验证成功率
            "val/presale_success_rate",
            "val/aftersale_success_rate",
        ],
        "episode": [
            "episode/success_rate",
            "episode/reward/mean",
            "episode/reward/max",
            "episode/length/mean",
            "episode/valid_action_ratio",
            # 按场景分的训练成功率
            "episode/presale_success_rate",
            "episode/aftersale_success_rate",
        ],
        "training": [
            "training/epoch",
            "training/global_step",
        ],
        "actor": [
            "actor/pg_loss",
            "actor/kl_loss",
            "actor/entropy_loss",
            "actor/ppo_kl",
        ],
        "critic": [
            "critic/rewards/mean",
            "critic/score/mean",
            "critic/advantages/mean",
        ],
        "performance": [
            "perf/throughput",
            "perf/mfu/actor",
            "perf/max_memory_allocated_gb",
        ],
        # 分段成功率（使用 target_len 避免幸存者偏差）
        "success_by_length": [
            "episode/success_rate/target_len_1_5",
            "episode/success_rate/target_len_6_10",
            "episode/success_rate/target_len_11_15",
            "episode/success_rate/target_len_16_20",
        ],
        # 验证集分段成功率
        "val_success_by_length": [
            "val/success_rate/target_len_1_5",
            "val/success_rate/target_len_6_10",
            "val/success_rate/target_len_11_15",
            "val/success_rate/target_len_16_20",
        ],
    }

    # 读取训练指标
    if log_path.exists():
        try:
            from tensorboard.backend.event_processing import event_file_loader
            event_files = sorted(log_path.glob("events.out.tfevents.*"), reverse=True)

            if event_files:
                loader = event_file_loader.EventFileLoader(str(event_files[0]))
                events = list(loader.Load())

                # 收集所有指标
                all_metrics = {}
                for event in events:
                    if event.HasField("summary"):
                        for value in event.summary.value:
                            tag = value.tag
                            # 获取值
                            val = None
                            if value.HasField("simple_value"):
                                val = value.simple_value
                            elif value.HasField("tensor"):
                                tensor = value.tensor
                                if tensor.float_val:
                                    val = tensor.float_val[0]
                                elif tensor.double_val:
                                    val = tensor.double_val[0]

                            if val is not None:
                                if tag not in all_metrics:
                                    all_metrics[tag] = []
                                all_metrics[tag].append({
                                    "step": event.step,
                                    "value": val,
                                })

                            if event.step:
                                result["current_step"] = event.step

                # 按分类整理指标
                for category, tags in key_metrics.items():
                    for tag in tags:
                        if tag in all_metrics:
                            values = all_metrics[tag]
                            result[category][tag] = {
                                "current": values[-1]["value"] if values else None,
                                "history": values,  # 返回历史用于绘图
                            }

                # 提取当前epoch
                if "training/epoch" in all_metrics:
                    result["current_epoch"] = int(all_metrics["training/epoch"][-1]["value"])

        except ImportError:
            result["error"] = "tensorboard not installed"
        except Exception as e:
            result["error"] = str(e)

    # 返回最近采样的playbook
    sample_file = log_path / "recent_playbooks.json"
    if sample_file.exists():
        try:
            with open(sample_file) as f:
                result["recent_playbooks"] = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    return result