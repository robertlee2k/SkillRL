#!/usr/bin/env python
"""
Audit script for detecting "Ghost Actions" in playbooks.

Ghost Action: An action that exists in `available_skills` list but not in `transitions` keys.

This is a read-only analysis script that does NOT modify any files.

Usage:
    python scripts/audit_skill_mismatch.py --playbook_path outputs/playbooks_all_fixed_v2.json
"""

import argparse
import json
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple
from pathlib import Path


def load_playbooks(path: str) -> List[Dict[str, Any]]:
    """Load playbooks from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_ghost_actions_in_node(node: Dict[str, Any]) -> List[str]:
    """
    Find ghost actions in a single node.

    Returns list of actions that are in available_skills but not in transitions.
    """
    available_skills = set(node.get('available_skills', []))
    transitions = set(node.get('transitions', {}).keys())

    # Ghost actions = available - transitions
    ghost_actions = available_skills - transitions

    return list(ghost_actions)


def audit_playbooks(playbooks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform comprehensive audit on all playbooks.

    Returns a dict with audit statistics.
    """
    # Overall counters
    total_playbooks = len(playbooks)
    total_nodes = 0
    nodes_with_ghost = 0
    playbooks_with_ghost = set()

    # Ghost action frequency
    ghost_action_counter = Counter()

    # Scenario breakdown
    scenario_ghost_counts = defaultdict(lambda: {'playbooks': set(), 'nodes': 0, 'actions': Counter()})

    # Detailed findings (for sampling)
    sample_findings = []

    for playbook in playbooks:
        playbook_id = playbook.get('playbook_id', 'unknown')
        scenario = playbook.get('scenario', 'unknown')
        nodes = playbook.get('nodes', {})

        playbook_has_ghost = False

        for node_id, node in nodes.items():
            total_nodes += 1

            ghost_actions = find_ghost_actions_in_node(node)

            if ghost_actions:
                nodes_with_ghost += 1
                playbook_has_ghost = True

                for action in ghost_actions:
                    ghost_action_counter[action] += 1
                    scenario_ghost_counts[scenario]['actions'][action] += 1

                scenario_ghost_counts[scenario]['nodes'] += 1

                # Collect sample (first 10 for each unique ghost action pattern)
                if len(sample_findings) < 50:
                    sample_findings.append({
                        'playbook_id': playbook_id,
                        'scenario': scenario,
                        'node_id': node_id,
                        'ghost_actions': ghost_actions,
                        'available_skills': node.get('available_skills', []),
                        'transitions': list(node.get('transitions', {}).keys())
                    })

        if playbook_has_ghost:
            playbooks_with_ghost.add(playbook_id)
            scenario_ghost_counts[scenario]['playbooks'].add(playbook_id)

    # Compile results
    results = {
        'total_playbooks': total_playbooks,
        'total_nodes': total_nodes,
        'playbooks_with_ghost': len(playbooks_with_ghost),
        'nodes_with_ghost': nodes_with_ghost,
        'ghost_action_frequency': ghost_action_counter,
        'scenario_breakdown': {
            scenario: {
                'playbooks_affected': len(data['playbooks']),
                'nodes_affected': data['nodes'],
                'top_actions': data['actions'].most_common(5)
            }
            for scenario, data in scenario_ghost_counts.items()
        },
        'sample_findings': sample_findings[:10]
    }

    return results


def print_report(results: Dict[str, Any]) -> None:
    """Print a formatted audit report."""
    print("=" * 70)
    print("                    PLAYBOOK GHOST ACTION AUDIT REPORT")
    print("=" * 70)
    print()

    # Section 1: Overall Statistics
    print("┌" + "─" * 68 + "┐")
    print("│  SECTION 1: OVERALL IMPACT                                          │")
    print("├" + "─" * 68 + "┤")

    total_playbooks = results['total_playbooks']
    total_nodes = results['total_nodes']
    playbooks_with_ghost = results['playbooks_with_ghost']
    nodes_with_ghost = results['nodes_with_ghost']

    playbook_pct = (playbooks_with_ghost / total_playbooks * 100) if total_playbooks > 0 else 0
    node_pct = (nodes_with_ghost / total_nodes * 100) if total_nodes > 0 else 0

    print(f"│  Total Playbooks Analyzed:     {total_playbooks:>8,}                          │")
    print(f"│  Total Nodes Analyzed:         {total_nodes:>8,}                          │")
    print("├" + "─" * 68 + "┤")
    print(f"│  Playbooks with Ghost Actions: {playbooks_with_ghost:>8,} ({playbook_pct:>5.1f}%)                 │")
    print(f"│  Nodes with Ghost Actions:     {nodes_with_ghost:>8,} ({node_pct:>5.1f}%)                 │")
    print("└" + "─" * 68 + "┘")
    print()

    # Section 2: Ghost Action Frequency Distribution
    print("┌" + "─" * 68 + "┐")
    print("│  SECTION 2: GHOST ACTION FREQUENCY DISTRIBUTION (CRITICAL)          │")
    print("├" + "─" * 68 + "┤")

    ghost_freq = results['ghost_action_frequency']
    total_ghost_occurrences = sum(ghost_freq.values())

    print(f"│  Total Ghost Action Occurrences: {total_ghost_occurrences:,}")
    print("├" + "─" * 68 + "┤")
    print("│  Rank │ Action                    │ Count    │ Percentage            │")
    print("├" + "─" * 68 + "┤")

    for rank, (action, count) in enumerate(ghost_freq.most_common(15), 1):
        pct = (count / total_ghost_occurrences * 100) if total_ghost_occurrences > 0 else 0
        print(f"│  {rank:>4} │ {action:<25} │ {count:>8,} │ {pct:>6.2f}%              │")

    print("└" + "─" * 68 + "┘")
    print()

    # Section 3: Scenario Breakdown
    print("┌" + "─" * 68 + "┐")
    print("│  SECTION 3: SCENARIO DISTRIBUTION                                   │")
    print("├" + "─" * 68 + "┤")
    print("│  Scenario    │ Playbooks │ Nodes   │ Top Ghost Actions             │")
    print("├" + "─" * 68 + "┤")

    for scenario, data in sorted(results['scenario_breakdown'].items()):
        top_actions = ', '.join([f"{a}({c})" for a, c in data['top_actions'][:3]])
        if len(top_actions) > 35:
            top_actions = top_actions[:32] + "..."
        print(f"│  {scenario:<11} │ {data['playbooks_affected']:>8,} │ {data['nodes_affected']:>7,} │ {top_actions:<30} │")

    print("└" + "─" * 68 + "┘")
    print()

    # Section 4: Sample Findings
    print("┌" + "─" * 68 + "┐")
    print("│  SECTION 4: SAMPLE FINDINGS (First 5)                               │")
    print("├" + "─" * 68 + "┤")

    for i, finding in enumerate(results['sample_findings'][:5], 1):
        print(f"│  [{i}] Playbook: {finding['playbook_id']:<30}    │")
        print(f"│      Scenario: {finding['scenario']:<20} Node: {finding['node_id']:<15}│")
        print(f"│      Ghost Actions: {str(finding['ghost_actions']):<47}│")
        print(f"│      Available: {str(finding['available_skills'][:5])[:47]}...│")
        print(f"│      Transitions: {str(finding['transitions'][:5])[:45]}...│")
        print("├" + "─" * 68 + "┤")

    print("└" + "─" * 68 + "┘")
    print()

    # Summary
    print("=" * 70)
    print("                         AUDIT SUMMARY")
    print("=" * 70)

    top_ghost = ghost_freq.most_common(1)
    top_action_name = top_ghost[0][0] if top_ghost else "N/A"
    top_action_count = top_ghost[0][1] if top_ghost else 0

    print(f"""
  ✓ Analyzed {total_playbooks:,} playbooks with {total_nodes:,} nodes
  ✓ Found {playbooks_with_ghost:,} playbooks ({playbook_pct:.1f}%) containing ghost actions
  ✓ Found {nodes_with_ghost:,} nodes ({node_pct:.1f}%) containing ghost actions
  ✓ Most frequent ghost action: '{top_action_name}' ({top_action_count:,} occurrences)

  ⚠ RECOMMENDATION: Review the top ghost actions above. Consider either:
    1. Adding missing transitions for these actions, OR
    2. Removing these actions from available_skills lists
""")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Audit ghost actions in playbooks')
    parser.add_argument('--playbook_path', type=str, default='outputs/playbooks_all_fixed_v2.json',
                        help='Path to playbooks JSON file')

    args = parser.parse_args()

    # Load and audit
    print(f"Loading playbooks from {args.playbook_path}...")
    playbooks = load_playbooks(args.playbook_path)
    print(f"Loaded {len(playbooks)} playbooks")
    print()

    results = audit_playbooks(playbooks)
    print_report(results)

    # Optionally save detailed results
    output_path = args.playbook_path.replace('.json', '_ghost_audit.json')
    if output_path == args.playbook_path:
        output_path = 'outputs/ghost_audit_results.json'

    with open(output_path, 'w', encoding='utf-8') as f:
        # Convert Counter and set to serializable types
        serializable_results = {
            'total_playbooks': results['total_playbooks'],
            'total_nodes': results['total_nodes'],
            'playbooks_with_ghost': results['playbooks_with_ghost'],
            'nodes_with_ghost': results['nodes_with_ghost'],
            'ghost_action_frequency': dict(results['ghost_action_frequency']),
            'scenario_breakdown': results['scenario_breakdown'],
        }
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    main()