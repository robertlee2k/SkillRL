#!/usr/bin/env python3
"""
Playbook Quality Audit: Compare raw sessions against generated playbooks.

Evaluates playbook quality across multiple dimensions:
1. Structural integrity (tree shape, branch count, reachability)
2. Skill mapping accuracy (does the "golden path" match the real conversation?)
3. Slot extraction quality (are key business facts captured?)
4. Sentiment plausibility (does sentiment match conversation tone?)
5. RL training fitness (is this playbook useful for RL?)

Usage:
    python scripts/audit_playbook_quality.py \
        --raw_data data_sample_100.json \
        --playbook_path outputs/playbooks_all_fixed_v2.json \
        --output_path outputs/playbook_audit_report.json
"""

import os
import sys
import json
import argparse
import logging
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.customer_service_env import VALID_SKILLS, SAFE_FALLBACK_SKILLS
from etl.aggregator import aggregate_turns
from etl.cleaner import clean_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. Structural Audit
# ============================================================

def audit_structure(pb: Dict) -> Dict[str, Any]:
    """Audit the tree structure of a playbook."""
    nodes = pb.get("nodes", {})
    issues = []

    # Basic counts
    total_nodes = len(nodes)
    non_terminal = {k: v for k, v in nodes.items() if k != "terminal"}

    # Branch counts
    branch_counts = []
    linear_nodes = []  # nodes with only 1 transition (excluding terminal/fallback)
    for nid, node in non_terminal.items():
        trans = node.get("transitions", {})
        n_branches = len(trans)
        branch_counts.append(n_branches)
        if n_branches <= 1 and not nid.startswith("fallback") and not nid.startswith("node_neg"):
            linear_nodes.append(nid)

    # Reachability (BFS from root)
    reachable = set()
    queue = ["root"]
    while queue:
        nid = queue.pop(0)
        if nid in reachable:
            continue
        reachable.add(nid)
        node = nodes.get(nid, {})
        for target in node.get("transitions", {}).values():
            if target not in reachable:
                queue.append(target)
        fb = node.get("default_fallback")
        if fb and fb not in reachable:
            queue.append(fb)

    unreachable = set(nodes.keys()) - reachable
    if unreachable - {"terminal"}:
        issues.append(f"unreachable_nodes: {unreachable - {'terminal'}}")

    # Depth (longest path from root)
    def max_depth(nid, visited=None):
        if visited is None:
            visited = set()
        if nid in visited or nid not in nodes or nid == "terminal":
            return 0
        visited.add(nid)
        children = list(nodes.get(nid, {}).get("transitions", {}).values())
        if not children:
            return 1
        return 1 + max(max_depth(c, visited.copy()) for c in children)

    depth = max_depth("root")

    # Has angry sentiment node?
    has_angry = any(
        n.get("sentiment") == "angry"
        for n in nodes.values()
    )

    # Has negative branch?
    has_neg_branch = any(nid.startswith("node_neg") for nid in nodes)

    avg_branches = sum(branch_counts) / len(branch_counts) if branch_counts else 0

    if not has_angry:
        issues.append("no_angry_sentiment_node")
    if len(linear_nodes) > len(non_terminal) * 0.5:
        issues.append(f"mostly_linear: {len(linear_nodes)}/{len(non_terminal)} nodes have <=1 branch")

    return {
        "total_nodes": total_nodes,
        "depth": depth,
        "avg_branches": round(avg_branches, 2),
        "linear_nodes": len(linear_nodes),
        "has_angry_node": has_angry,
        "has_neg_branch": has_neg_branch,
        "unreachable_count": len(unreachable - {"terminal"}),
        "issues": issues,
    }


# ============================================================
# 2. Golden Path Alignment
# ============================================================

def reconstruct_golden_path(pb: Dict) -> List[str]:
    """
    Reconstruct the "golden path" — the sequence of transitions
    that follows the main storyline (root → node1 → node2 → ... → terminal).
    """
    nodes = pb.get("nodes", {})
    path = []
    current = "root"
    visited = set()

    while current and current != "terminal" and current not in visited:
        visited.add(current)
        node = nodes.get(current, {})
        transitions = node.get("transitions", {})

        # Find the "main" transition (typically the first non-negative one)
        main_skill = None
        main_target = None
        for skill, target in transitions.items():
            if not target.startswith("node_neg") and not target.startswith("fallback"):
                main_skill = skill
                main_target = target
                break

        if main_skill:
            path.append(main_skill)
            current = main_target
        else:
            # No main path found, try any transition
            if transitions:
                skill, target = next(iter(transitions.items()))
                path.append(skill)
                current = target
            else:
                break

    return path


def audit_skill_mapping(pb: Dict, raw_session: Dict) -> Dict[str, Any]:
    """
    Audit how well the playbook's golden path maps to the real conversation.
    """
    golden_path = reconstruct_golden_path(pb)

    # Reconstruct what the real agent did from raw messages
    messages = raw_session.get("messages", [])
    agent_actions_raw = []
    for msg in messages:
        if msg.get("sent_by") in ("ASSISTANT", "QA", "QA_VENDOR"):
            content = msg.get("content", "")
            if content and content != "[IMAGE]":
                agent_actions_raw.append(content[:100])

    # Count skill categories used
    skill_categories = Counter()
    for skill in golden_path:
        prefix = skill.split("_")[0]
        skill_categories[prefix] += 1

    # Check scenario alignment
    scenario = pb.get("scenario", "unknown")
    expected_prefixes = {
        "presale": {"pre", "gen"},
        "logistics": {"log", "gen"},
        "aftersale": {"aft", "gen"},
    }
    expected = expected_prefixes.get(scenario, set())
    misaligned_skills = [
        s for s in golden_path
        if s.split("_")[0] not in expected and s.split("_")[0] != "gen"
    ]

    issues = []
    if misaligned_skills:
        issues.append(f"scenario_skill_mismatch: {misaligned_skills} in {scenario}")

    # Check if golden path length is reasonable vs raw conversation
    raw_agent_turns = len(agent_actions_raw)
    path_len = len(golden_path)
    if raw_agent_turns > 0 and abs(path_len - raw_agent_turns) > raw_agent_turns * 0.5:
        issues.append(
            f"path_length_mismatch: golden={path_len} vs raw_agent_turns={raw_agent_turns}"
        )

    return {
        "golden_path": golden_path,
        "golden_path_length": path_len,
        "raw_agent_turns": raw_agent_turns,
        "skill_categories": dict(skill_categories),
        "misaligned_skills": misaligned_skills,
        "issues": issues,
    }


# ============================================================
# 3. Slot Quality Audit
# ============================================================

def audit_slots(pb: Dict, raw_session: Dict) -> Dict[str, Any]:
    """Audit slot extraction quality."""
    nodes = pb.get("nodes", {})
    issues = []

    # Collect all slot updates across the tree
    all_slots = {}
    slot_dimensions_used = set()
    empty_slot_nodes = 0
    total_non_terminal = 0

    for nid, node in nodes.items():
        if nid == "terminal":
            continue
        total_non_terminal += 1
        updates = node.get("slot_updates", {})
        if not updates:
            empty_slot_nodes += 1
        for key, val in updates.items():
            if val and str(val).strip():
                slot_dimensions_used.add(key)
                all_slots[key] = val

    # Check the 5 expected dimensions
    expected_dims = {"user_intent", "item_specifics", "system_status",
                     "agent_commitments", "other_crucial_context"}
    missing_dims = expected_dims - slot_dimensions_used

    # Check if user_intent is captured (most critical)
    has_user_intent = "user_intent" in slot_dimensions_used

    # Check raw session for key business facts
    raw_messages = raw_session.get("messages", [])
    raw_text = " ".join(m.get("content", "") for m in raw_messages if m.get("content"))

    # Simple heuristic: check if order-related info is captured
    has_order = raw_session.get("has_order", False)
    order_amount = raw_session.get("order_amount", 0)
    pb_has_order = pb.get("business_outcome", {}).get("has_order", False)
    pb_order_amount = pb.get("business_outcome", {}).get("order_amount", 0)

    if has_order != pb_has_order:
        issues.append(f"order_mismatch: raw={has_order} vs pb={pb_has_order}")
    if has_order and order_amount is not None and pb_order_amount is not None and abs(order_amount - pb_order_amount) > 1:
        issues.append(f"amount_mismatch: raw={order_amount} vs pb={pb_order_amount}")

    empty_ratio = empty_slot_nodes / total_non_terminal if total_non_terminal > 0 else 0

    if not has_user_intent:
        issues.append("missing_user_intent")
    if empty_ratio > 0.7:
        issues.append(f"sparse_slots: {empty_slot_nodes}/{total_non_terminal} nodes have empty slots")

    return {
        "dimensions_used": sorted(slot_dimensions_used),
        "dimensions_missing": sorted(missing_dims),
        "has_user_intent": has_user_intent,
        "empty_slot_ratio": round(empty_ratio, 2),
        "business_outcome_match": has_order == pb_has_order,
        "issues": issues,
    }


# ============================================================
# 4. Buyer Text Quality
# ============================================================

def audit_buyer_text(pb: Dict, raw_session: Dict) -> Dict[str, Any]:
    """Audit buyer_text quality: real vs hallucinated."""
    nodes = pb.get("nodes", {})
    raw_messages = raw_session.get("messages", [])

    # Extract real buyer messages
    real_buyer_texts = set()
    for msg in raw_messages:
        if msg.get("sent_by") == "BUYER":
            content = msg.get("content", "").strip()
            if content and content != "[IMAGE]":
                real_buyer_texts.add(content)

    # Check each node's buyer_text
    real_count = 0
    hallucinated_count = 0
    empty_count = 0
    issues = []

    for nid, node in nodes.items():
        if nid == "terminal":
            continue
        bt = node.get("buyer_text", "").strip()
        if not bt or bt == "[END]":
            empty_count += 1
            continue

        # Check if this buyer_text appears in real messages
        is_real = any(bt in real or real in bt for real in real_buyer_texts)
        if is_real:
            real_count += 1
        else:
            hallucinated_count += 1

    total = real_count + hallucinated_count + empty_count
    hallucination_ratio = hallucinated_count / total if total > 0 else 0

    # High hallucination ratio is expected (branches are hallucinated)
    # but the root and main path should use real text
    root_bt = nodes.get("root", {}).get("buyer_text", "")
    root_is_real = any(root_bt in real or real in root_bt for real in real_buyer_texts)
    if not root_is_real and root_bt:
        issues.append("root_buyer_text_not_from_real_session")

    return {
        "real_buyer_texts": real_count,
        "hallucinated_buyer_texts": hallucinated_count,
        "empty_buyer_texts": empty_count,
        "hallucination_ratio": round(hallucination_ratio, 2),
        "root_text_is_real": root_is_real,
        "issues": issues,
    }


# ============================================================
# 5. RL Training Fitness
# ============================================================

def audit_rl_fitness(pb: Dict) -> Dict[str, Any]:
    """Audit how useful this playbook is for RL training."""
    nodes = pb.get("nodes", {})
    issues = []

    rl_steps = pb.get("rl_steps", 0)
    scenario = pb.get("scenario", "unknown")

    # Check if there's meaningful reward signal
    has_order = pb.get("business_outcome", {}).get("has_order", False)

    # Count unique skills across all transitions
    all_skills = set()
    for node in nodes.values():
        for skill in node.get("transitions", {}).keys():
            all_skills.add(skill)

    # Check skill diversity
    skill_diversity = len(all_skills)

    # Check if there are both positive and negative paths
    sentiments = [n.get("sentiment", "neutral") for n in nodes.values()]
    has_positive_path = any(s in ("calm", "happy") for s in sentiments)
    has_negative_path = any(s == "angry" for s in sentiments)

    # Check transition consistency: every transition target should exist
    dangling_refs = []
    for nid, node in nodes.items():
        for skill, target in node.get("transitions", {}).items():
            if target not in nodes:
                dangling_refs.append(f"{nid}.{skill}->{target}")

    if dangling_refs:
        issues.append(f"dangling_transitions: {dangling_refs}")

    # Check if too short for meaningful learning
    if rl_steps <= 1:
        issues.append("too_short: rl_steps<=1, minimal learning signal")

    # Check if scenario is unknown
    if scenario == "unknown":
        issues.append("unknown_scenario: reward function may not apply correctly")

    if not has_negative_path:
        issues.append("no_negative_path: no angry sentiment for punishment signal")

    if skill_diversity < 3:
        issues.append(f"low_skill_diversity: only {skill_diversity} unique skills")

    # Overall fitness score (0-10)
    score = 10
    if rl_steps <= 1:
        score -= 3
    if not has_negative_path:
        score -= 2
    if skill_diversity < 3:
        score -= 2
    if scenario == "unknown":
        score -= 1
    if dangling_refs:
        score -= 2
    score = max(0, score)

    return {
        "rl_steps": rl_steps,
        "skill_diversity": skill_diversity,
        "unique_skills": sorted(all_skills),
        "has_positive_path": has_positive_path,
        "has_negative_path": has_negative_path,
        "dangling_transitions": len(dangling_refs),
        "fitness_score": score,
        "issues": issues,
    }


# ============================================================
# Main Audit
# ============================================================

def audit_single(pb: Dict, raw_session: Optional[Dict] = None) -> Dict[str, Any]:
    """Run all audits on a single playbook."""
    result = {
        "playbook_id": pb.get("playbook_id", ""),
        "session_id": pb.get("session_id", ""),
        "scenario": pb.get("scenario", ""),
        "rl_steps": pb.get("rl_steps", 0),
    }

    result["structure"] = audit_structure(pb)
    result["rl_fitness"] = audit_rl_fitness(pb)

    if raw_session:
        result["skill_mapping"] = audit_skill_mapping(pb, raw_session)
        result["slots"] = audit_slots(pb, raw_session)
        result["buyer_text"] = audit_buyer_text(pb, raw_session)

    # Collect all issues
    all_issues = []
    for key in ["structure", "rl_fitness", "skill_mapping", "slots", "buyer_text"]:
        if key in result and "issues" in result[key]:
            for issue in result[key]["issues"]:
                all_issues.append(f"[{key}] {issue}")

    result["all_issues"] = all_issues
    result["issue_count"] = len(all_issues)
    result["fitness_score"] = result["rl_fitness"]["fitness_score"]

    return result


def main():
    parser = argparse.ArgumentParser(description="Playbook quality audit")
    parser.add_argument("--raw_data", default="/home/bo.li/data/SkillRL/session_order_converted.json")
    parser.add_argument("--playbook_path", default="outputs/playbooks_all_fixed_v2.json")
    parser.add_argument("--output_path", default="outputs/playbook_audit_report.json")
    parser.add_argument("--full_audit", action="store_true",
                        help="Audit ALL playbooks (structure + RL fitness only for those without raw data)")
    args = parser.parse_args()

    # Load raw data
    logger.info(f"Loading raw sessions from {args.raw_data}")
    with open(args.raw_data, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    # Support both formats: direct list or {sessions: [...]}
    if isinstance(raw_data, list):
        sessions = raw_data
    else:
        sessions = raw_data.get("sessions", [])
    session_map = {s["session_id"]: s for s in sessions}
    logger.info(f"Loaded {len(sessions)} raw sessions")

    # Load playbooks
    logger.info(f"Loading playbooks from {args.playbook_path}")
    with open(args.playbook_path, "r", encoding="utf-8") as f:
        playbooks = json.load(f)
    logger.info(f"Loaded {len(playbooks)} playbooks")

    # Run audits
    paired_results = []  # playbooks with raw session match
    unpaired_results = []  # playbooks without raw session

    for pb in playbooks:
        sid = pb.get("session_id", "")
        raw = session_map.get(sid)

        if raw:
            result = audit_single(pb, raw)
            paired_results.append(result)
        elif args.full_audit:
            result = audit_single(pb)
            unpaired_results.append(result)

    logger.info(f"Paired audits (with raw data): {len(paired_results)}")
    logger.info(f"Unpaired audits (structure only): {len(unpaired_results)}")

    all_results = paired_results + unpaired_results

    # ============================================================
    # Aggregate Statistics
    # ============================================================

    # Structure stats (all playbooks)
    all_for_struct = all_results if args.full_audit else paired_results
    struct_stats = {
        "avg_nodes": round(sum(r["structure"]["total_nodes"] for r in all_for_struct) / len(all_for_struct), 1),
        "avg_depth": round(sum(r["structure"]["depth"] for r in all_for_struct) / len(all_for_struct), 1),
        "avg_branches": round(sum(r["structure"]["avg_branches"] for r in all_for_struct) / len(all_for_struct), 2),
        "pct_has_angry": round(sum(1 for r in all_for_struct if r["structure"]["has_angry_node"]) / len(all_for_struct) * 100, 1),
        "pct_has_neg_branch": round(sum(1 for r in all_for_struct if r["structure"]["has_neg_branch"]) / len(all_for_struct) * 100, 1),
        "pct_mostly_linear": round(sum(1 for r in all_for_struct if any("mostly_linear" in i for i in r["structure"]["issues"])) / len(all_for_struct) * 100, 1),
    }

    # RL fitness stats
    fitness_scores = [r["fitness_score"] for r in all_for_struct]
    fitness_stats = {
        "avg_score": round(sum(fitness_scores) / len(fitness_scores), 1),
        "score_distribution": dict(Counter(fitness_scores)),
        "avg_rl_steps": round(sum(r["rl_steps"] for r in all_for_struct) / len(all_for_struct), 1),
        "avg_skill_diversity": round(sum(r["rl_fitness"]["skill_diversity"] for r in all_for_struct) / len(all_for_struct), 1),
        "pct_no_negative_path": round(sum(1 for r in all_for_struct if not r["rl_fitness"]["has_negative_path"]) / len(all_for_struct) * 100, 1),
    }

    # Paired-only stats
    paired_stats = {}
    if paired_results:
        # Skill mapping
        total_misaligned = sum(len(r.get("skill_mapping", {}).get("misaligned_skills", [])) for r in paired_results)
        path_length_mismatches = sum(
            1 for r in paired_results
            if any("path_length_mismatch" in i for i in r.get("skill_mapping", {}).get("issues", []))
        )

        # Slot quality
        has_intent = sum(1 for r in paired_results if r.get("slots", {}).get("has_user_intent", False))
        biz_match = sum(1 for r in paired_results if r.get("slots", {}).get("business_outcome_match", False))
        avg_empty_ratio = sum(r.get("slots", {}).get("empty_slot_ratio", 0) for r in paired_results) / len(paired_results)

        # Buyer text
        avg_hallucination = sum(r.get("buyer_text", {}).get("hallucination_ratio", 0) for r in paired_results) / len(paired_results)
        root_real = sum(1 for r in paired_results if r.get("buyer_text", {}).get("root_text_is_real", False))

        paired_stats = {
            "skill_mapping": {
                "total_misaligned_skills": total_misaligned,
                "pct_path_length_mismatch": round(path_length_mismatches / len(paired_results) * 100, 1),
            },
            "slot_quality": {
                "pct_has_user_intent": round(has_intent / len(paired_results) * 100, 1),
                "pct_business_outcome_match": round(biz_match / len(paired_results) * 100, 1),
                "avg_empty_slot_ratio": round(avg_empty_ratio, 2),
            },
            "buyer_text": {
                "avg_hallucination_ratio": round(avg_hallucination, 2),
                "pct_root_text_is_real": round(root_real / len(paired_results) * 100, 1),
            },
        }

    # Top issues
    issue_counter = Counter()
    for r in all_for_struct:
        for issue in r.get("all_issues", []):
            # Normalize issue for counting
            category = issue.split("]")[0] + "]" if "]" in issue else issue
            issue_type = issue.split(":")[0] if ":" in issue else issue
            issue_counter[issue_type] += 1

    # ============================================================
    # Build Report
    # ============================================================

    report = {
        "summary": {
            "total_playbooks": len(playbooks),
            "paired_with_raw": len(paired_results),
            "unpaired_audited": len(unpaired_results),
        },
        "structure_stats": struct_stats,
        "rl_fitness_stats": fitness_stats,
        "paired_comparison_stats": paired_stats,
        "top_issues": dict(issue_counter.most_common(20)),
        "worst_playbooks": sorted(
            [{"id": r["playbook_id"], "score": r["fitness_score"], "issues": r["all_issues"]}
             for r in all_for_struct if r["fitness_score"] <= 5],
            key=lambda x: x["score"]
        )[:20],
        "detailed_paired_results": paired_results,
    }

    # ============================================================
    # Print Summary
    # ============================================================

    print("\n" + "=" * 70)
    print("PLAYBOOK QUALITY AUDIT REPORT")
    print("=" * 70)
    print(f"Total playbooks:        {len(playbooks)}")
    print(f"Paired with raw data:   {len(paired_results)}")

    print(f"\n--- Structure (n={len(all_for_struct)}) ---")
    for k, v in struct_stats.items():
        print(f"  {k:25s}: {v}")

    print(f"\n--- RL Fitness (n={len(all_for_struct)}) ---")
    print(f"  avg_fitness_score:       {fitness_stats['avg_score']}/10")
    print(f"  avg_rl_steps:            {fitness_stats['avg_rl_steps']}")
    print(f"  avg_skill_diversity:     {fitness_stats['avg_skill_diversity']}")
    print(f"  pct_no_negative_path:    {fitness_stats['pct_no_negative_path']}%")

    if paired_stats:
        print(f"\n--- Skill Mapping (n={len(paired_results)}) ---")
        for k, v in paired_stats["skill_mapping"].items():
            print(f"  {k:30s}: {v}")

        print(f"\n--- Slot Quality (n={len(paired_results)}) ---")
        for k, v in paired_stats["slot_quality"].items():
            print(f"  {k:30s}: {v}")

        print(f"\n--- Buyer Text (n={len(paired_results)}) ---")
        for k, v in paired_stats["buyer_text"].items():
            print(f"  {k:30s}: {v}")

    print(f"\n--- Top Issues ---")
    for issue, count in issue_counter.most_common(15):
        print(f"  {issue:50s}: {count}")

    print("=" * 70)

    # Save
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Report saved to {args.output_path}")


if __name__ == "__main__":
    main()
