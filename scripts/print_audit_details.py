#!/usr/bin/env python3
"""Print detailed audit results."""
import json

with open("outputs/playbook_audit_report.json") as f:
    report = json.load(f)

# Fitness score distribution
print("=== Fitness Score Distribution ===")
dist = report["rl_fitness_stats"]["score_distribution"]
for score in sorted(dist.keys(), key=lambda x: int(x)):
    count = dist[score]
    print(f"  Score {score}: {count} playbooks")

# Worst playbooks
print("\n=== Worst Playbooks (score <= 5) ===")
for wp in report["worst_playbooks"][:10]:
    pid = wp["id"]
    sc = wp["score"]
    print(f"  {pid}: score={sc}")
    for iss in wp["issues"]:
        print(f"    - {iss}")

# Scenario skill mismatch
print("\n=== Scenario-Skill Mismatch Examples ===")
count = 0
for r in report["detailed_paired_results"]:
    sm = r.get("skill_mapping", {})
    mis = sm.get("misaligned_skills", [])
    if mis and count < 8:
        pid = r["playbook_id"]
        scenario = r["scenario"]
        gp = sm["golden_path"]
        print(f"  {pid} ({scenario}): misaligned={mis}")
        print(f"    golden_path={gp}")
        count += 1

# Root not real
print("\n=== Root Buyer Text Not Real ===")
for r in report["detailed_paired_results"]:
    bt = r.get("buyer_text", {})
    if not bt.get("root_text_is_real", True):
        pid = r["playbook_id"]
        issues = r.get("all_issues", [])
        print(f"  {pid}: {[i for i in issues if 'buyer_text' in i]}")

# Slot quality details
print("\n=== Slot Quality: Missing Dimensions ===")
dim_counter = {}
for r in report["detailed_paired_results"]:
    sl = r.get("slots", {})
    for dim in sl.get("dimensions_missing", []):
        dim_counter[dim] = dim_counter.get(dim, 0) + 1
for dim, cnt in sorted(dim_counter.items(), key=lambda x: -x[1]):
    print(f"  {dim}: missing in {cnt}/98 playbooks")

# Path length mismatch details
print("\n=== Path Length Mismatch Examples ===")
count = 0
for r in report["detailed_paired_results"]:
    sm = r.get("skill_mapping", {})
    issues = sm.get("issues", [])
    for iss in issues:
        if "path_length_mismatch" in iss and count < 8:
            pid = r["playbook_id"]
            gp_len = sm["golden_path_length"]
            raw_turns = sm["raw_agent_turns"]
            print(f"  {pid}: golden_path={gp_len} steps, raw_agent={raw_turns} turns")
            count += 1
