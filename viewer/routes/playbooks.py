"""Playbook 路由"""
from typing import Optional
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse

from ..data import data_loader
from ..templates_config import templates
from ..config import PAGE_SIZE
from ..models import TurnStats

router = APIRouter()


def get_unique_values() -> dict:
    """获取筛选器的唯一值"""
    playbooks = data_loader.playbooks
    scenarios = sorted(set(p.scenario for p in playbooks))
    subtypes = sorted(set(p.subtype for p in playbooks))
    sentiments = sorted(set(
        node.sentiment
        for p in playbooks
        for node in p.nodes.values()
    ))

    # 获取 rl_steps 范围
    steps_values = [p.rl_steps for p in playbooks if p.rl_steps is not None]
    max_steps = max(steps_values) if steps_values else 20

    return {
        "scenarios": scenarios,
        "subtypes": subtypes,
        "sentiments": sentiments,
        "max_rl_steps": max_steps
    }


def compute_turn_stats() -> TurnStats:
    """计算轮次统计信息"""
    playbooks = data_loader.playbooks

    steps_values = [p.rl_steps for p in playbooks if p.rl_steps is not None]
    turns_values = [p.effective_turn_count for p in playbooks if p.effective_turn_count is not None]

    if not steps_values:
        return TurnStats(
            total=len(playbooks),
            with_turn_info=0,
            min_turns=0, max_turns=0, avg_turns=0,
            min_steps=0, max_steps=0, avg_steps=0,
            p90_steps=0, p95_steps=0, p99_steps=0,
            over_20_steps=0,
            steps_distribution={}
        )

    # 计算 percentiles
    sorted_steps = sorted(steps_values)
    n = len(sorted_steps)
    p90 = sorted_steps[int(n * 0.90)]
    p95 = sorted_steps[int(n * 0.95)]
    p99 = sorted_steps[int(n * 0.99)]

    # 计算 distribution
    from collections import Counter
    steps_dist = dict(Counter(steps_values))

    return TurnStats(
        total=len(playbooks),
        with_turn_info=len(steps_values),
        min_turns=min(turns_values) if turns_values else 0,
        max_turns=max(turns_values) if turns_values else 0,
        avg_turns=sum(turns_values) / len(turns_values) if turns_values else 0,
        min_steps=min(steps_values),
        max_steps=max(steps_values),
        avg_steps=sum(steps_values) / len(steps_values),
        p90_steps=p90,
        p95_steps=p95,
        p99_steps=p99,
        over_20_steps=sum(1 for s in steps_values if s > 20),
        steps_distribution=steps_dist
    )


@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    page: int = Query(1, ge=1),
    search: Optional[str] = Query(None),
    scenario: Optional[str] = Query(None),
    subtype: Optional[str] = Query(None),
    has_order: Optional[str] = Query(None),
    sentiment: Optional[str] = Query(None),
    rl_steps_max: Optional[int] = Query(None),  # 新增：rl_steps 过滤
):
    """首页 - Playbook 列表（支持URL参数）"""
    playbooks, total = filter_playbooks(
        search=search,
        scenario=scenario,
        subtype=subtype,
        has_order=has_order,
        sentiment=sentiment,
        rl_steps_max=rl_steps_max,
        page=page,
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "playbooks": playbooks,
            "total": total,
            "page": page,
            "total_pages": (total + PAGE_SIZE - 1) // PAGE_SIZE,
            "filters": get_unique_values(),
            "turn_stats": compute_turn_stats(),
            "current_filters": {
                "search": search or "",
                "scenario": scenario or "",
                "subtype": subtype or "",
                "has_order": has_order or "",
                "sentiment": sentiment or "",
                "rl_steps_max": rl_steps_max or "",
            }
        }
    )


@router.get("/api/playbooks-table", response_class=HTMLResponse)
async def playbooks_table(
    request: Request,
    page: int = Query(1, ge=1),
    search: Optional[str] = Query(None),
    scenario: Optional[str] = Query(None),
    subtype: Optional[str] = Query(None),
    has_order: Optional[str] = Query(None),
    sentiment: Optional[str] = Query(None),
    rl_steps_max: Optional[int] = Query(None),
):
    """获取 playbook 列表表格HTML（HTMX用）"""
    playbooks, total = filter_playbooks(
        search=search,
        scenario=scenario,
        subtype=subtype,
        has_order=has_order,
        sentiment=sentiment,
        rl_steps_max=rl_steps_max,
        page=page,
    )

    return templates.TemplateResponse(
        "_playbooks_table.html",
        {
            "request": request,
            "playbooks": playbooks,
            "total": total,
            "page": page,
            "total_pages": (total + PAGE_SIZE - 1) // PAGE_SIZE,
        }
    )


def filter_playbooks(
    search: Optional[str],
    scenario: Optional[str],
    subtype: Optional[str],
    has_order: Optional[str],
    sentiment: Optional[str],
    rl_steps_max: Optional[int],
    page: int,
) -> tuple[list, int]:
    """筛选 playbook"""
    playbooks = data_loader.playbooks

    # 搜索
    if search:
        search_lower = search.lower()
        playbooks = [
            p for p in playbooks
            if search_lower in p.playbook_id.lower()
            or search_lower in p.session_id.lower()
            or search_lower in p.scenario.lower()
        ]

    # scenario 筛选
    if scenario:
        playbooks = [p for p in playbooks if p.scenario == scenario]

    # subtype 筛选
    if subtype:
        playbooks = [p for p in playbooks if p.subtype == subtype]

    # has_order 筛选
    if has_order == "true":
        playbooks = [p for p in playbooks if p.business_outcome.has_order]
    elif has_order == "false":
        playbooks = [p for p in playbooks if not p.business_outcome.has_order]

    # sentiment 筛选（检查所有节点）
    if sentiment:
        playbooks = [
            p for p in playbooks
            if any(node.sentiment == sentiment for node in p.nodes.values())
        ]

    # rl_steps 筛选（新增）
    if rl_steps_max is not None:
        playbooks = [
            p for p in playbooks
            if p.rl_steps is not None and p.rl_steps <= rl_steps_max
        ]

    total = len(playbooks)
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE

    return playbooks[start:end], total


@router.get("/playbook/{playbook_id}", response_class=HTMLResponse)
async def playbook_detail(request: Request, playbook_id: str):
    """Playbook 详情页"""
    playbook = data_loader.get_playbook(playbook_id)
    if not playbook:
        return HTMLResponse(content="Playbook not found", status_code=404)

    session = data_loader.get_session(playbook.session_id)

    return templates.TemplateResponse(
        "playbook_detail.html",
        {
            "request": request,
            "playbook": playbook,
            "session": session,
        }
    )


@router.get("/api/playbook/{playbook_id}")
async def get_playbook(playbook_id: str):
    """获取 playbook 数据 (API)"""
    playbook = data_loader.get_playbook(playbook_id)
    if not playbook:
        return {"error": "not found"}
    return playbook.model_dump()


@router.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """获取 session 数据 (API)"""
    session = data_loader.get_session(session_id)
    if not session:
        return {"error": "not found"}
    return session.model_dump()