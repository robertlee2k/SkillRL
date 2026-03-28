"""Playbook Viewer 配置"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据文件路径
PLAYBOOKS_PATH = PROJECT_ROOT / "outputs" / "playbooks_all.json"
SESSIONS_PATH = PROJECT_ROOT / "session_order_converted.json"

# 分页配置
PAGE_SIZE = 50