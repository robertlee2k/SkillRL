#!/usr/bin/env python3
"""
Content-level playbook quality audit.

Answers the core question: Is what the playbook teaches the model
actually correct and useful for real customer service?

Dimensions:
1. Skill mapping accuracy: Does the assigned skill match what the agent actually did?
2. Transition logic: Do the "correct" paths make business sense?
3. Information loss: What critical info from the real conversation is missing?
4. Action space coverage: Are the 31 skills sufficient for real conversations?
5. Reward signal quality: Will the RL reward guide the model correctly?

Usage:
    python scripts/audit_content_quality.py \
        --raw_data /home/bo.li/data/SkillRL/session_order_converted.json \
        --playbook_path outputs/playbooks_all_fixed_v2.json \
        --output_path outputs/content_audit_report.json
"""

import os
import sys
import json
import re
import argparse
import logging
import random
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.customer_service_env import VALID_SKILLS, SAFE_FALLBACK_SKILLS, HIGH_RISK_SKILLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# Skill-to-action semantic mapping
# ============================================================

SKILL_KEYWORDS = {
    # General
    "gen_greet": ["你好", "欢迎", "亲", "您好", "在的", "很高兴"],
    "gen_empathize": ["理解", "抱歉", "不好意思", "辛苦", "麻烦", "心情", "着急"],
    "gen_clarify": ["请问", "确认", "是否", "哪个", "什么意思", "能否说明", "具体"],
    "gen_verify_order": ["订单号", "订单", "核实", "查一下", "帮您查", "单号"],
    "gen_hold": ["稍等", "请等", "马上", "帮您看", "查询中"],
    "gen_transfer": ["转接", "人工", "专员", "同事", "升级"],
    "gen_apologize": ["抱歉", "对不起", "不好意思", "给您带来", "非常抱歉"],
    "gen_close": ["感谢", "祝您", "再见", "满意", "好评", "有问题随时"],
    # Presale
    "pre_query_product": ["这款", "商品", "产品", "宝贝"],
    "pre_check_stock": ["库存", "有货", "缺货", "补货", "现货"],
    "pre_compare": ["对比", "区别", "哪个好", "推荐哪", "差别"],
    "pre_recommend": ["推荐", "建议", "适合", "搭配"],
    "pre_answer_spec": ["规格", "尺寸", "材质", "颜色", "参数", "功能", "怎么用", "机洗", "洗涤"],
    "pre_check_promo": ["优惠", "活动", "折扣", "券", "满减", "赠品", "促销"],
    "pre_guide_purchase": ["下单", "购买", "拍", "付款", "链接"],
    # Logistics
    "log_query_status": ["物流", "快递", "发货", "到哪了", "运输"],
    "log_query_detail": ["单号", "物流详情", "跟踪", "查询"],
    "log_estimate_arrival": ["到货", "几天", "什么时候到", "预计"],
    "log_modify_address": ["地址", "修改地址", "改地址"],
    "log_contact_courier": ["快递员", "联系快递", "派送"],
    "log_delay_notify": ["延迟", "慢", "催", "加急"],
    "log_lost_claim": ["丢件", "没收到", "丢失"],
    # Aftersale
    "aft_check_policy": ["退货", "退款", "换货", "政策", "规则", "七天", "无理由", "运费险"],
    "aft_collect_evidence": ["照片", "图片", "证据", "拍照", "截图"],
    "aft_initiate_refund": ["退款", "退钱"],
    "aft_initiate_return": ["退货", "寄回"],
    "aft_initiate_exchange": ["换货", "更换"],
    "aft_schedule_pickup": ["上门取件", "取件", "寄回"],
    "aft_track_progress": ["进度", "处理到哪", "审核"],
    "aft_compensate": ["补偿", "赔偿", "差价"],
    "aft_reject_explain": ["无法", "不能", "不支持", "超过期限"],
}


def extract_agent_turns(raw_session: Dict) -> List[Dict]:
    """Extract agent turns with their preceding buyer context."""
    messages = raw_session.get("messages", [])
    turns = []
    prev_buyer_text = ""

    for msg in messages:
        sent_by = msg.get("sent_by", "")
        content = msg.get("content", "")
        fmt = msg.get("format", "TEXT")

        if sent_by == "BUYER":
            if fmt == "TEXT" and content:
                prev_buyer_text = content
            elif fmt == "IMAGE":
                prev_buyer_text = "[发送了图片]"
        elif sent_by in ("ASSISTANT", "QA", "QA_VENDOR"):
            if fmt == "TEXT" and content and content != "[IMAGE]":
                turns.append({
                    "buyer_context": prev_buyer_text,
                    "agent_response": content,
                })
    return turns


def guess_skill_from_text(agent_text: str, buyer_text: str = "") -> List[str]:
    """Guess which skill(s) an agent response corresponds to."""
    combined = (agent_text + " " + buyer_text).lower()
    matches = []
    for skill, keywords in SKILL_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score > 0:
            matches.append((skill, score))
    matches.sort(key=lambda x: -x[1])
    return [m[0] for m in matches[:3]]  # top 3 guesses


def audit_single_content(pb: Dict, raw_session: Dict) -> Dict[str, Any]:
    """Deep content audit of a single playbook vs its raw session."""
    nodes = pb.get("nodes", {})
    scenario = pb.get("scenario", "unknown")
    pb_id = pb.get("playbook_id", "")

    # Extract real agent turns
    real_turns = extract_agent_turns(raw_session)
    raw_messages = raw_session.get("messages", [])

    issues = []

    # ============================================================
    # 1. Skill mapping semantic check
    # ============================================================
    # Walk the golden path and check if assigned skills match agent behavior
    golden_skills = []
    current = "root"
    visited = set()
    skill_match_results = []

    turn_idx = 0
    while current and current != "terminal" and current not in visited:
        visited.add(current)
        node = nodes.get(current, {})
        transitions = node.get("transitions", {})

        # Find main transition
        main_skill = None
        main_target = None
        for skill, target in transitions.items():
            if not target.startswith("node_neg") and not target.startswith("fallback"):
                main_skill = skill
                main_target = target
                break

        if main_skill and turn_idx < len(real_turns):
            turn = real_turns[turn_idx]
            guessed = guess_skill_from_text(turn["agent_response"], turn["buyer_context"])
            is_plausible = main_skill in guessed or len(guessed) == 0

            skill_match_results.append({
                "step": turn_idx,
                "assigned_skill": main_skill,
                "guessed_skills": guessed,
                "plausible": is_plausible,
                "agent_text_preview": turn["agent_response"][:80],
            })
            golden_skills.append(main_skill)
            turn_idx += 1

        if main_target:
            current = main_target
        else:
            break

    plausible_count = sum(1 for r in skill_match_results if r["plausible"])
    total_checked = len(skill_match_results)
    skill_accuracy = plausible_count / total_checked if total_checked > 0 else 0

    # ============================================================
    # 2. Information loss analysis
    # ============================================================
    # Check what buyer said vs what playbook captured
    real_buyer_texts = []
    for msg in raw_messages:
        if msg.get("sent_by") == "BUYER" and msg.get("format") == "TEXT":
            content = msg.get("content", "").strip()
            if content:
                real_buyer_texts.append(content)

    pb_buyer_texts = []
    for nid, node in nodes.items():
        bt = node.get("buyer_text", "").strip()
        if bt and bt != "[END]":
            pb_buyer_texts.append(bt)

    # Check for key buyer concerns not captured
    # Simple heuristic: look for question marks and key phrases in real data
    buyer_questions = [t for t in real_buyer_texts if "?" in t or "？" in t or "吗" in t or "呢" in t]
    captured_questions = 0
    for q in buyer_questions:
        if any(q in pbt or pbt in q for pbt in pb_buyer_texts):
            captured_questions += 1

    question_capture_rate = captured_questions / len(buyer_questions) if buyer_questions else 1.0

    # ============================================================
    # 3. Agent response coverage
    # ============================================================
    # How many real agent responses are represented in the playbook?
    real_agent_count = len(real_turns)
    pb_path_length = len(golden_skills)
    coverage_ratio = pb_path_length / real_agent_count if real_agent_count > 0 else 0

    # ============================================================
    # 4. Action space fitness
    # ============================================================
    # Check if real agent actions can be mapped to the 31 skills
    unmappable_turns = []
    for i, turn in enumerate(real_turns):
        guessed = guess_skill_from_text(turn["agent_response"], turn["buyer_context"])
        if not guessed:
            unmappable_turns.append({
                "turn_idx": i,
                "agent_text": turn["agent_response"][:100],
                "buyer_context": turn["buyer_context"][:80],
            })

    unmappable_rate = len(unmappable_turns) / len(real_turns) if real_turns else 0

    # ============================================================
    # 5. Reward signal analysis
    # ============================================================
    has_order = pb.get("business_outcome", {}).get("has_order", False)
    order_amount = pb.get("business_outcome", {}).get("order_amount", 0)

    # Check if the playbook's terminal state makes sense
    # For presale: won=True should correlate with has_order
    # For aftersale: won=True means issue resolved
    reward_concern = None
    if scenario == "presale" and not has_order:
        # Presale without order - the "won" reward is just 2.0 (no order bonus)
        # This is fine but worth noting
        reward_concern = "presale_no_order"
    elif scenario == "unknown":
        reward_concern = "unknown_scenario_weak_reward"

    # ============================================================
    # 6. Transition logic check
    # ============================================================
    # Check if transitions make business sense
    logic_issues = []
    for nid, node in nodes.items():
        if nid == "terminal":
            continue
        transitions = node.get("transitions", {})
        buyer_text = node.get("buyer_text", "").lower()

        # Check: if buyer asks about refund, is aft_check_policy or aft_initiate_refund available?
        if any(kw in buyer_text for kw in ["退款", "退货", "换货"]):
            has_aft = any(s.startswith("aft_") for s in transitions)
            if not has_aft and scenario in ("aftersale", "logistics"):
                logic_issues.append(f"{nid}: buyer asks about return/refund but no aft_ skill in transitions")

        # Check: if buyer asks about shipping, is log_ skill available?
        if any(kw in buyer_text for kw in ["物流", "快递", "发货", "到哪"]):
            has_log = any(s.startswith("log_") for s in transitions)
            if not has_log and scenario == "logistics":
                logic_issues.append(f"{nid}: buyer asks about logistics but no log_ skill in transitions")

    # ============================================================
    # Compile result
    # ============================================================
    if skill_accuracy < 0.5 and total_checked >= 3:
        issues.append(f"low_skill_accuracy: {plausible_count}/{total_checked}")
    if coverage_ratio < 0.5 and real_agent_count >= 4:
        issues.append(f"low_coverage: pb_path={pb_path_length} vs real_turns={real_agent_count}")
    if question_capture_rate < 0.5 and len(buyer_questions) >= 2:
        issues.append(f"buyer_questions_lost: captured {captured_questions}/{len(buyer_questions)}")
    if unmappable_rate > 0.3:
        issues.append(f"high_unmappable_rate: {len(unmappable_turns)}/{len(real_turns)}")
    if logic_issues:
        issues.append(f"transition_logic: {len(logic_issues)} issues")

    return {
        "playbook_id": pb_id,
        "scenario": scenario,
        "rl_steps": pb.get("rl_steps", 0),
        "real_agent_turns": real_agent_count,
        "real_buyer_questions": len(buyer_questions),
        "skill_mapping": {
            "accuracy": round(skill_accuracy, 2),
            "checked": total_checked,
            "plausible": plausible_count,
        },
        "information_loss": {
            "coverage_ratio": round(coverage_ratio, 2),
            "question_capture_rate": round(question_capture_rate, 2),
            "buyer_questions_total": len(buyer_questions),
            "buyer_questions_captured": captured_questions,
        },
        "action_space": {
            "unmappable_rate": round(unmappable_rate, 2),
            "unmappable_count": len(unmappable_turns),
            "unmappable_examples": unmappable_turns[:3],
        },
        "reward_concern": reward_concern,
        "transition_logic_issues": len(logic_issues),
        "issues": issues,
        "issue_count": len(issues),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="/home/bo.li/data/SkillRL/session_order_converted.json")
    parser.add_argument("--playbook_path", default="outputs/playbooks_all_fixed_v2.json")
    parser.add_argument("--output_path", default="outputs/content_audit_report.json")
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading raw sessions from {args.raw_data}")
    with open(args.raw_data, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    sessions = raw_data if isinstance(raw_data, list) else raw_data.get("sessions", [])
    session_map = {s["session_id"]: s for s in sessions}
    logger.info(f"Loaded {len(sessions)} raw sessions")

    logger.info(f"Loading playbooks from {args.playbook_path}")
    with open(args.playbook_path, "r", encoding="utf-8") as f:
        playbooks = json.load(f)
    logger.info(f"Loaded {len(playbooks)} playbooks")

    # Run audit
    results = []
    for pb in playbooks:
        sid = pb.get("session_id", "")
        raw = session_map.get(sid)
        if raw:
            result = audit_single_content(pb, raw)
            results.append(result)

    logger.info(f"Audited {len(results)} playbooks")

    # ============================================================
    # Aggregate
    # ============================================================
    total = len(results)

    # Skill accuracy distribution
    acc_buckets = {">=0.8": 0, "0.5-0.8": 0, "0.3-0.5": 0, "<0.3": 0}
    for r in results:
        acc = r["skill_mapping"]["accuracy"]
        if acc >= 0.8:
            acc_buckets[">=0.8"] += 1
        elif acc >= 0.5:
            acc_buckets["0.5-0.8"] += 1
        elif acc >= 0.3:
            acc_buckets["0.3-0.5"] += 1
        else:
            acc_buckets["<0.3"] += 1

    avg_skill_acc = sum(r["skill_mapping"]["accuracy"] for r in results) / total
    avg_coverage = sum(r["information_loss"]["coverage_ratio"] for r in results) / total
    avg_question_capture = sum(r["information_loss"]["question_capture_rate"] for r in results) / total
    avg_unmappable = sum(r["action_space"]["unmappable_rate"] for r in results) / total

    # Coverage distribution
    cov_buckets = {">=0.8": 0, "0.5-0.8": 0, "0.3-0.5": 0, "<0.3": 0}
    for r in results:
        cov = r["information_loss"]["coverage_ratio"]
        if cov >= 0.8:
            cov_buckets[">=0.8"] += 1
        elif cov >= 0.5:
            cov_buckets["0.5-0.8"] += 1
        elif cov >= 0.3:
            cov_buckets["0.3-0.5"] += 1
        else:
            cov_buckets["<0.3"] += 1

    # By scenario
    scenario_stats = defaultdict(lambda: {
        "count": 0, "acc_sum": 0, "cov_sum": 0, "issues_sum": 0
    })
    for r in results:
        s = r["scenario"]
        scenario_stats[s]["count"] += 1
        scenario_stats[s]["acc_sum"] += r["skill_mapping"]["accuracy"]
        scenario_stats[s]["cov_sum"] += r["information_loss"]["coverage_ratio"]
        scenario_stats[s]["issues_sum"] += r["issue_count"]

    scenario_summary = {}
    for s, st in scenario_stats.items():
        n = st["count"]
        scenario_summary[s] = {
            "count": n,
            "avg_skill_accuracy": round(st["acc_sum"] / n, 2),
            "avg_coverage": round(st["cov_sum"] / n, 2),
            "avg_issues": round(st["issues_sum"] / n, 1),
        }

    # Unmappable action examples (aggregate)
    all_unmappable = []
    for r in results:
        for ex in r["action_space"]["unmappable_examples"]:
            all_unmappable.append(ex)
    unmappable_sample = random.sample(all_unmappable, min(20, len(all_unmappable)))

    # Issue distribution
    issue_counter = Counter()
    for r in results:
        for iss in r["issues"]:
            tag = iss.split(":")[0]
            issue_counter[tag] += 1

    # Reward concerns
    reward_concerns = Counter(r["reward_concern"] for r in results if r["reward_concern"])

    # Problematic playbooks
    problematic = [r for r in results if r["issue_count"] >= 3]

    # ============================================================
    # Report
    # ============================================================
    report = {
        "summary": {
            "total_audited": total,
            "avg_skill_accuracy": round(avg_skill_acc, 2),
            "avg_coverage_ratio": round(avg_coverage, 2),
            "avg_question_capture_rate": round(avg_question_capture, 2),
            "avg_unmappable_rate": round(avg_unmappable, 2),
            "playbooks_with_issues": sum(1 for r in results if r["issue_count"] > 0),
            "playbooks_with_3plus_issues": len(problematic),
        },
        "skill_accuracy_distribution": acc_buckets,
        "coverage_distribution": cov_buckets,
        "by_scenario": scenario_summary,
        "issue_distribution": dict(issue_counter.most_common(20)),
        "reward_concerns": dict(reward_concerns),
        "unmappable_action_samples": unmappable_sample,
        "worst_playbooks": sorted(
            [{"id": r["playbook_id"], "scenario": r["scenario"],
              "skill_acc": r["skill_mapping"]["accuracy"],
              "coverage": r["information_loss"]["coverage_ratio"],
              "issues": r["issues"]}
             for r in results if r["issue_count"] >= 2],
            key=lambda x: x["skill_acc"]
        )[:30],
    }

    # ============================================================
    # Print
    # ============================================================
    print("\n" + "=" * 70)
    print("CONTENT QUALITY AUDIT REPORT")
    print("=" * 70)
    print(f"Total audited:              {total}")
    print(f"Avg skill accuracy:         {avg_skill_acc:.0%}")
    print(f"Avg coverage ratio:         {avg_coverage:.0%}")
    print(f"Avg question capture rate:  {avg_question_capture:.0%}")
    print(f"Avg unmappable action rate: {avg_unmappable:.0%}")
    print(f"Playbooks with issues:      {report['summary']['playbooks_with_issues']} ({report['summary']['playbooks_with_issues']/total*100:.1f}%)")
    print(f"Playbooks with 3+ issues:   {len(problematic)} ({len(problematic)/total*100:.1f}%)")

    print(f"\n--- Skill Accuracy Distribution ---")
    for bucket, count in acc_buckets.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {bucket:8s}: {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\n--- Coverage Distribution ---")
    for bucket, count in cov_buckets.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {bucket:8s}: {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\n--- By Scenario ---")
    for s, st in sorted(scenario_summary.items()):
        print(f"  {s:12s}: n={st['count']:4d}, skill_acc={st['avg_skill_accuracy']:.0%}, coverage={st['avg_coverage']:.0%}, avg_issues={st['avg_issues']}")

    print(f"\n--- Issue Distribution ---")
    for issue, count in issue_counter.most_common(10):
        print(f"  {issue:40s}: {count:5d} ({count/total*100:.1f}%)")

    print(f"\n--- Reward Concerns ---")
    for concern, count in reward_concerns.most_common():
        print(f"  {concern:40s}: {count}")

    print(f"\n--- Unmappable Action Samples (agent did something outside 31 skills) ---")
    for ex in unmappable_sample[:10]:
        print(f"  buyer: {ex['buyer_context'][:60]}")
        print(f"  agent: {ex['agent_text'][:80]}")
        print()

    print("=" * 70)

    # Save
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Report saved to {args.output_path}")


if __name__ == "__main__":
    main()
