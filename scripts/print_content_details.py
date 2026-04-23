#!/usr/bin/env python3
import json

with open("outputs/content_audit_report.json") as f:
    report = json.load(f)

print("=== Unmappable Action Samples ===")
for ex in report.get("unmappable_action_samples", []):
    bc = ex.get("buyer_context", "")
    at = ex.get("agent_text", "")
    print(f"  buyer: {bc[:80]}")
    print(f"  agent: {at[:100]}")
    print()

print("=== Worst Playbooks ===")
for wp in report.get("worst_playbooks", [])[:15]:
    pid = wp.get("id", "")
    sc = wp.get("scenario", "")
    sa = wp.get("skill_acc", 0)
    cov = wp.get("coverage", 0)
    print(f"  {pid} ({sc}): skill_acc={sa}, coverage={cov}")
    for iss in wp.get("issues", []):
        print(f"    - {iss}")

print("\n=== Issue Distribution ===")
for iss, cnt in report.get("issue_distribution", {}).items():
    total = report["summary"]["total_audited"]
    print(f"  {iss:45s}: {cnt:5d} ({cnt/total*100:.1f}%)")

print("\n=== By Scenario ===")
for sc, st in sorted(report.get("by_scenario", {}).items()):
    print(f"  {sc:12s}: n={st['count']:4d}, skill_acc={st['avg_skill_accuracy']:.0%}, coverage={st['avg_coverage']:.0%}")
