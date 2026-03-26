"""FastAPI 应用入口"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="Playbook Viewer")

# 获取当前文件所在目录
BASE_DIR = Path(__file__).parent

# 静态文件
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.on_event("startup")
async def startup():
    """启动时预加载数据"""
    from .data import data_loader
    data_loader.load()


# 注册路由（放在 startup 之后避免循环导入）
from .routes import playbooks
app.include_router(playbooks.router)