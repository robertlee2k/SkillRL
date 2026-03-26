"""Playbook 路由"""
from typing import Optional
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse

from ..data import data_loader
from ..templates_config import templates
from ..config import PAGE_SIZE

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
    return {"scenarios": scenarios, "subtypes": subtypes, "sentiments": sentiments}


@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    page: int = Query(1, ge=1),
    search: Optional[str] = Query(None),
    scenario: Optional[str] = Query(None),
    subtype: Optional[str] = Query(None),
    has_order: Optional[str] = Query(None),
    sentiment: Optional[str] = Query(None),
):
    """首页 - Playbook 列表（支持URL参数）"""
    playbooks, total = filter_playbooks(
        search=search,
        scenario=scenario,
        subtype=subtype,
        has_order=has_order,
        sentiment=sentiment,
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
            "current_filters": {
                "search": search or "",
                "scenario": scenario or "",
                "subtype": subtype or "",
                "has_order": has_order or "",
                "sentiment": sentiment or "",
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
):
    """获取 playbook 列表表格HTML（HTMX用）"""
    playbooks, total = filter_playbooks(
        search=search,
        scenario=scenario,
        subtype=subtype,
        has_order=has_order,
        sentiment=sentiment,
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

    total = len(playbooks)
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE

    return playbooks[start:end], total