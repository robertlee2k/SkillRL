#!/usr/bin/env python
"""
Recover playbook_id from bad cases by reverse lookup.

This script extracts buyer messages from bad case files and matches them
against playbooks to recover the lost playbook_id metadata.

Usage:
    python scripts/recover_bad_cases.py \
        --bad_case_file outputs/bad_cases/bad_cases_step90_*.json \
        --playbook_path outputs/playbooks_all.json \
        --output_path outputs/recovered_bad_cases_step90.json
"""

import os
import sys
import json
import re
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from glob import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def extract_buyer_message(task: str) -> Optional[str]:
    """
    Extract buyer message from the task field.

    The task field contains the full decoded prompt in ChatML format.
    We need to extract the text between '## 买家消息\n' and '\n\n## 任务'.
    """
    # Primary pattern: ## 买家消息\n{buyer_text}\n\n## 任务
    match = re.search(r'## 买家消息\n(.*?)\n\n## 任务', task, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: try to find user turn and extract buyer message
    user_match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', task, re.DOTALL)
    if user_match:
        user_content = user_match.group(1)
        buyer_match = re.search(r'## 买家消息\n(.*?)(?=\n##|\n<\|im_end\|>|$)', user_content, re.DOTALL)
        if buyer_match:
            return buyer_match.group(1).strip()

    return None


def build_buyer_to_playbook_mapping(playbooks: List[Dict]) -> Dict[str, Dict]:
    """
    Build a mapping from buyer_text to playbook metadata.

    Returns:
        Dict mapping buyer_text to {'playbook_id': str, 'scenario': str}
    """
    mapping = {}

    for p in playbooks:
        playbook_id = p.get('playbook_id', '')
        scenario = p.get('scenario', 'unknown')
        nodes = p.get('nodes', {})

        # Get buyer_text from root node
        if isinstance(nodes, dict) and 'root' in nodes:
            root = nodes['root']
            if isinstance(root, dict) and 'buyer_text' in root:
                buyer_text = root['buyer_text']
                # Store first occurrence (or we could store all)
                if buyer_text not in mapping:
                    mapping[buyer_text] = {
                        'playbook_id': playbook_id,
                        'scenario': scenario
                    }

    logger.info(f"[Mapping] Built {len(mapping)} buyer_text -> playbook entries")
    return mapping


def parse_action_from_text(action_text: str) -> str:
    """Extract action skill_id from text like '<action>gen_clarify</action>'."""
    match = re.search(r'<action>(.*?)</action>', action_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try to extract action from the text
    lines = action_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('action:') or 'gen_' in line or 'pre_' in line or 'log_' in line or 'aft_' in line:
            # Extract the skill_id
            skill_match = re.search(r'(gen_\w+|pre_\w+|log_\w+|aft_\w+)', line)
            if skill_match:
                return skill_match.group(1)
    return action_text[:50]


def parse_reasoning_from_text(text: str) -> str:
    """Extract reasoning from text like '<think>...</think>' or similar thinking block."""
    # Look for <think>...</think> pattern (most common in our model)
    match = re.search(r'<think>\s*\n?(.*?)\n?</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Look for ◃...◃ pattern
    match = re.search(r'◃(.*?)◃', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Look for ⋄⋄⋄...⋄⋄⋄ pattern
    match = re.search(r'⋄⋄⋄(.*?)⋄⋄⋄', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Look for ◈...◈ pattern
    match = re.search(r'◈(.*?)◈', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Look for thinking block with ```
    match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Look for analysis in brackets
    match = re.search(r'〔(.*?)〕', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Look for 【...】
    match = re.search(r'【(.*?)】', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return ''


def convert_trajectory_to_dialogue_history(trajectory: List[Dict]) -> List[Dict]:
    """
    Convert trajectory format to dialogue_history format.

    Input trajectory format:
    [{'action': '...', 'observation': '...'}, ...]

    Output dialogue_history format:
    [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}, ...]

    Note: The first step usually contains the system prompt, so we skip it.
    """
    dialogue_history = []

    # Skip the first step if it looks like a system prompt
    start_idx = 0
    if trajectory and trajectory[0].get('observation', '').startswith('system\n'):
        start_idx = 1

    for step in trajectory[start_idx:]:
        observation = step.get('observation', '')
        action = step.get('action', '')

        if observation:
            dialogue_history.append({
                'role': 'user',
                'content': observation[:500]  # Truncate for readability
            })

        if action:
            dialogue_history.append({
                'role': 'assistant',
                'content': action[:500]
            })

    return dialogue_history


def extract_error_actions(trajectory: List[Dict]) -> List[Dict]:
    """
    Extract error actions from trajectory.

    For now, we treat all actions in failed trajectories as potential errors.
    In the future, we could filter by checking if the action was invalid.
    """
    error_actions = []

    for i, step in enumerate(trajectory):
        action_text = step.get('action', '')
        observation = step.get('observation', '')

        if not action_text:
            continue

        action_id = parse_action_from_text(action_text)
        reasoning = parse_reasoning_from_text(action_text)

        # Check if there's an error indication in the observation
        system_message = ''
        if 'invalid' in observation.lower() or 'error' in observation.lower():
            system_message = observation[:200]

        error_actions.append({
            'step': i,
            'action': action_id,
            'reasoning': reasoning,
            'system_message': system_message
        })

    return error_actions


def recover_bad_cases(
    bad_case_file: str,
    playbook_path: str,
    output_path: str
) -> Dict[str, Any]:
    """
    Main recovery function.

    1. Load bad cases from file
    2. Load playbooks and build buyer_text mapping
    3. Match buyer messages to recover playbook_id
    4. Convert to standard format for evolve_skills.py
    """

    # Load bad cases
    logger.info(f"[Load] Loading bad cases from {bad_case_file}")
    with open(bad_case_file, 'r', encoding='utf-8') as f:
        bad_case_data = json.load(f)

    raw_bad_cases = bad_case_data.get('bad_cases', [])
    logger.info(f"[Load] Found {len(raw_bad_cases)} bad cases")

    # Load playbooks
    logger.info(f"[Load] Loading playbooks from {playbook_path}")
    with open(playbook_path, 'r', encoding='utf-8') as f:
        playbooks = json.load(f)

    # Build mapping
    buyer_mapping = build_buyer_to_playbook_mapping(playbooks)

    # Process each bad case
    recovered_cases = []
    unmatched_cases = []
    match_stats = {
        'total': len(raw_bad_cases),
        'matched': 0,
        'unmatched': 0,
        'by_scenario': {}
    }

    for i, bc in enumerate(raw_bad_cases):
        task = bc.get('task', '')
        trajectory = bc.get('trajectory', [])
        score = bc.get('score', 0)

        # Extract buyer message
        buyer_msg = extract_buyer_message(task)

        if not buyer_msg:
            logger.warning(f"[Match] Case {i}: Could not extract buyer message")
            unmatched_cases.append({
                'index': i,
                'reason': 'no_buyer_message',
                'task_preview': task[:200]
            })
            match_stats['unmatched'] += 1
            continue

        # Lookup playbook
        if buyer_msg in buyer_mapping:
            meta = buyer_mapping[buyer_msg]
            playbook_id = meta['playbook_id']
            scenario = meta['scenario']

            # Convert trajectory to dialogue history
            dialogue_history = convert_trajectory_to_dialogue_history(trajectory)

            # Extract error actions
            error_actions = extract_error_actions(trajectory)

            recovered_cases.append({
                'playbook_id': playbook_id,
                'scenario': scenario,
                'buyer_message': buyer_msg,
                'dialogue_history': dialogue_history,
                'error_actions': error_actions,
                'score': score,
                'original_index': i
            })

            match_stats['matched'] += 1
            match_stats['by_scenario'][scenario] = match_stats['by_scenario'].get(scenario, 0) + 1

        else:
            # No match found - could be due to buyer_message variations
            unmatched_cases.append({
                'index': i,
                'reason': 'no_playbook_match',
                'buyer_message': buyer_msg[:100],
                'task_preview': task[:200]
            })
            match_stats['unmatched'] += 1

    # Build output
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'source_file': bad_case_file,
        'recovery_stats': match_stats,
        'bad_cases': recovered_cases,
        'unmatched_cases': unmatched_cases[:50]  # Keep first 50 for debugging
    }

    # Save output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"[Save] Saved {len(recovered_cases)} recovered cases to {output_path}")
    logger.info(f"[Stats] Matched: {match_stats['matched']}/{match_stats['total']} "
                f"({match_stats['matched']/match_stats['total']*100:.1f}%)")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description='Recover playbook_id from bad cases by reverse lookup'
    )
    parser.add_argument(
        '--bad_case_file',
        type=str,
        default=None,
        help='Path to bad case JSON file (supports glob pattern)'
    )
    parser.add_argument(
        '--bad_case_dir',
        type=str,
        default='outputs/bad_cases',
        help='Directory containing bad case files (if --bad_case_file not specified)'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=None,
        help='Training step number to process (e.g., 90 for step90)'
    )
    parser.add_argument(
        '--playbook_path',
        type=str,
        default='outputs/playbooks_all.json',
        help='Path to playbooks JSON file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Output path for recovered bad cases'
    )

    args = parser.parse_args()

    # Determine bad case file
    if args.bad_case_file:
        bad_case_file = args.bad_case_file
    elif args.step is not None:
        # Find file matching step
        pattern = os.path.join(args.bad_case_dir, f'bad_cases_step{args.step}_*.json')
        matches = glob(pattern)
        if matches:
            bad_case_file = sorted(matches)[-1]  # Take the latest
            logger.info(f"[Find] Found file: {bad_case_file}")
        else:
            logger.error(f"No file found matching pattern: {pattern}")
            sys.exit(1)
    else:
        # Find the latest file
        pattern = os.path.join(args.bad_case_dir, 'bad_cases_step*.json')
        matches = glob(pattern)
        if matches:
            bad_case_file = sorted(matches)[-1]
            logger.info(f"[Find] Using latest file: {bad_case_file}")
        else:
            logger.error(f"No bad case files found in {args.bad_case_dir}")
            sys.exit(1)

    # Determine output path
    if args.output_path:
        output_path = args.output_path
    else:
        # Generate from input filename
        basename = os.path.basename(bad_case_file).replace('.json', '')
        output_path = f'outputs/recovered_{basename}.json'

    # Run recovery
    result = recover_bad_cases(
        bad_case_file=bad_case_file,
        playbook_path=args.playbook_path,
        output_path=output_path
    )

    # Print summary
    print("\n" + "=" * 60)
    print("RECOVERY SUMMARY")
    print("=" * 60)
    print(f"Source: {bad_case_file}")
    print(f"Output: {output_path}")
    print(f"Total cases: {result['recovery_stats']['total']}")
    print(f"Matched: {result['recovery_stats']['matched']}")
    print(f"Unmatched: {result['recovery_stats']['unmatched']}")
    print("\nBy scenario:")
    for scenario, count in sorted(result['recovery_stats']['by_scenario'].items()):
        print(f"  {scenario}: {count}")
    print("=" * 60)


if __name__ == '__main__':
    main()