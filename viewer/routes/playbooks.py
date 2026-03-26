"""Playbook 路由"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ..data import data_loader
from ..templates_config import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """首页 - Playbook 列表"""
    playbooks = data_loader.playbooks[:50]  # 先显示前50条
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "playbooks": playbooks}
    )