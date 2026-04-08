"""Playbook Viewer 配置"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据文件路径（使用已清理幽灵动作的版本）
PLAYBOOKS_PATH = PROJECT_ROOT / "outputs" / "playbooks_all_fixed.json"
SESSIONS_PATH = Path("/home/bo.li/data/SkillRL/session_order_converted.json")

# 分页配置
PAGE_SIZE = 50

# 沙盒配置
DEFAULT_CHECKPOINT_PATH = "/home/bo.li/data/SkillRL/skillrl_models/customer_service/step_216"
DEFAULT_PLAYBOOK_PATH = str(PROJECT_ROOT / "outputs" / "playbooks_all_fixed.json")