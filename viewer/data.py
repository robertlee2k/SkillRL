"""数据加载和索引"""
import json
from typing import Optional

from .config import PLAYBOOKS_PATH, SESSIONS_PATH
from .models import Playbook, Session


class DataLoader:
    """启动时加载数据，构建索引"""

    def __init__(self):
        self._playbooks: list[Playbook] = []
        self._playbook_by_id: dict[str, Playbook] = {}
        self._session_by_id: dict[str, Session] = {}
        self._loaded = False

    def load(self) -> None:
        """加载所有数据"""
        if self._loaded:
            return

        # 加载 playbooks
        if PLAYBOOKS_PATH.exists():
            with open(PLAYBOOKS_PATH, encoding="utf-8") as f:
                data = json.load(f)

            # 处理每个 playbook，计算缺失的 rl_steps
            processed_playbooks = []
            for item in data:
                # 如果 rl_steps 缺失，基于 effective_turn_count 计算
                if item.get('rl_steps') is None and item.get('effective_turn_count') is not None:
                    # rl_steps = User turns 数量 ≈ (total turns + 1) // 2
                    item['rl_steps'] = (item['effective_turn_count'] + 1) // 2

                processed_playbooks.append(Playbook(**item))

            self._playbooks = processed_playbooks
            self._playbook_by_id = {p.playbook_id: p for p in self._playbooks}

        # 加载 sessions
        if SESSIONS_PATH.exists():
            with open(SESSIONS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                session = Session(**item)
                self._session_by_id[session.session_id] = session

        self._loaded = True

    @property
    def playbooks(self) -> list[Playbook]:
        self.load()
        return self._playbooks

    def get_playbook(self, playbook_id: str) -> Optional[Playbook]:
        self.load()
        return self._playbook_by_id.get(playbook_id)

    def get_session(self, session_id: str) -> Optional[Session]:
        self.load()
        return self._session_by_id.get(session_id)

    def get_playbooks_by_session(self, session_id: str) -> list[Playbook]:
        """根据 session_id 查找关联的 playbooks"""
        self.load()
        return [p for p in self._playbooks if p.session_id == session_id]


# 全局单例
data_loader = DataLoader()