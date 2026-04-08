#!/usr/bin/env python
"""
清理幽灵动作：确保 available_skills 只包含 transitions 中的动作 + 安全话术

核心逻辑: available_skills = transitions.keys() | SAFE_FALLBACK_SKILLS

Usage:
    python scripts/fix_ghost_actions.py \
        --input outputs/playbooks_all.json \
        --output outputs/playbooks_all_fixed.json
"""

import json
import argparse
import logging
from typing import Dict, Any, Set, List, Tuple
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 安全话术：不推进业务但不会激怒买家
SAFE_FALLBACK_SKILLS = {'gen_clarify', 'gen_empathize', 'gen_greet', 'gen_apologize', 'gen_hold'}

# 业务强制绑定映射：当 slots 中存在特定槽位时，必须添加的技能和对应的目标节点
# 注意：这些技能必须同时在 available_skills 和 transitions 中添加
SLOT_BINDINGS = {
    'invoice_requested': {
        'skill': 'aft_issue_invoice',
        'default_target': 'terminal'  # 发票开具后通常结束对话
    },
    # 未来可扩展其他强制绑定技能
}


def fix_node(node: Dict[str, Any], node_id: str, slots: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Fix a single node by removing ghost business actions.

    Args:
        node: Node data dict
        node_id: Node identifier (for logging)
        slots: Current slots state (for dynamic patching)

    Returns:
        Fixed node dict
    """
    transitions = node.get('transitions', {})
    transition_keys = set(transitions.keys())

    # 基础修正: available_skills = transitions + 安全话术
    correct_available = transition_keys | SAFE_FALLBACK_SKILLS

    # 动态打补丁：检查 slots 中是否有需要强制绑定的技能
    if slots:
        for slot_name, binding in SLOT_BINDINGS.items():
            if slots.get(slot_name):
                skill = binding['skill']
                default_target = binding['default_target']

                # 添加到 available_skills
                correct_available.add(skill)

                # 关键：必须同时在 transitions 中添加！否则又变成幽灵动作
                if skill not in transition_keys:
                    transitions[skill] = default_target
                    logger.info(f"  [Patch] Added forced skill '{skill}' -> '{default_target}' (triggered by slot '{slot_name}')")

    # 更新 available_skills
    node['available_skills'] = list(correct_available)

    # 更新 transitions（可能有动态补丁）
    node['transitions'] = transitions

    return node


def fix_playbook(playbook: Dict[str, Any]) -> Tuple[Dict[str, Any], Counter, Counter]:
    """
    Fix all nodes in a playbook using DFS tree traversal.

    [CRITICAL] Playbook is a TREE structure! Must traverse from root,
    accumulating slots along the path to ensure child nodes inherit
    parent slot updates.

    Args:
        playbook: Playbook dict

    Returns:
        Tuple of (Fixed playbook dict, ghost_removed Counter, safe_kept Counter)
    """
    playbook = playbook.copy()
    nodes = playbook.get('nodes', {})

    # Get initial slots (base state for root node)
    initial_slots = playbook.get('initial_slots', {})

    fixed_nodes = {}
    ghost_removed = Counter()
    safe_kept = Counter()
    visited = set()  # Prevent cycles

    def dfs_fix_node(node_id: str, accumulated_slots: Dict[str, Any]) -> None:
        """
        DFS traversal: fix node and propagate slots to children.

        Args:
            node_id: Current node ID
            accumulated_slots: Slots accumulated from root to this node (inclusive)
        """
        if node_id in visited or node_id not in nodes:
            return

        visited.add(node_id)
        node = nodes[node_id].copy()

        # Merge current node's slot_updates into accumulated_slots
        node_slots = accumulated_slots.copy()
        node_slots.update(node.get('slot_updates', {}))

        # Stats before fix
        old_available = set(node.get('available_skills', []))
        old_transitions = set(node.get('transitions', {}).keys())

        # Execute fix with current accumulated slots
        fixed_node = fix_node(node, node_id, node_slots)

        # Stats after fix
        new_available = set(fixed_node.get('available_skills', []))
        new_transitions = set(fixed_node.get('transitions', {}).keys())

        # Track removed ghost actions
        removed = old_available - new_available
        for action in removed:
            if action in SAFE_FALLBACK_SKILLS:
                safe_kept[action] += 1
            else:
                ghost_removed[action] += 1

        fixed_nodes[node_id] = fixed_node

        # DFS: traverse to children via transitions edges
        transitions = fixed_node.get('transitions', {})
        for skill, target_node_id in transitions.items():
            if target_node_id and target_node_id != node_id:  # Avoid self-loops
                dfs_fix_node(target_node_id, node_slots)

    # Start DFS from root
    dfs_fix_node('root', initial_slots.copy())

    # Handle orphan nodes (not reachable from root) - still fix them
    for node_id in nodes:
        if node_id not in visited:
            node = nodes[node_id].copy()
            # Orphan nodes use initial_slots only (no accumulated context)
            fixed_node = fix_node(node, node_id, initial_slots.copy())
            fixed_nodes[node_id] = fixed_node

    playbook['nodes'] = fixed_nodes

    return playbook, ghost_removed, safe_kept


def main():
    parser = argparse.ArgumentParser(description='Fix ghost actions in playbooks')
    parser.add_argument('--input', required=True, help='Input playbooks JSON')
    parser.add_argument('--output', required=True, help='Output playbooks JSON')
    parser.add_argument('--dry-run', action='store_true', help='Only analyze, do not write output')

    args = parser.parse_args()

    logger.info(f"Loading playbooks from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        playbooks = json.load(f)

    logger.info(f"Total playbooks: {len(playbooks)}")

    # 统计
    total_ghost_removed = Counter()
    total_safe_kept = Counter()
    total_forced_skills_added = 0

    fixed_playbooks = []
    for pb in playbooks:
        fixed_pb, ghost_removed, safe_kept = fix_playbook(pb)
        fixed_playbooks.append(fixed_pb)
        total_ghost_removed.update(ghost_removed)
        total_safe_kept.update(safe_kept)

    # 输出统计
    logger.info("=" * 60)
    logger.info("FIX SUMMARY")
    logger.info("=" * 60)

    logger.info("\n[Ghost Actions Removed] (业务幽灵动作已移除):")
    for action, count in total_ghost_removed.most_common(20):
        logger.info(f"  {action}: {count}")

    logger.info(f"\n[Safe Actions Kept] (安全话术已保留):")
    for action, count in total_safe_kept.most_common():
        logger.info(f"  {action}: {count}")

    logger.info(f"\nTotal ghost actions removed: {sum(total_ghost_removed.values())}")
    logger.info(f"Total safe actions kept: {sum(total_safe_kept.values())}")

    if args.dry_run:
        logger.info("\n[DRY RUN] No output file written")
        return

    # 写入输出
    logger.info(f"\nWriting fixed playbooks to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(fixed_playbooks, f, ensure_ascii=False, indent=2)

    logger.info(f"Done! Fixed {len(fixed_playbooks)} playbooks")


if __name__ == '__main__':
    main()