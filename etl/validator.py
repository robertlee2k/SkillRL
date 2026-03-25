# etl/validator.py
"""Playbook validation per design spec Section 6.1"""
from typing import List, Tuple


# 31 valid skills from design spec
VALID_SKILLS = {
    # General skills
    'gen_greet', 'gen_empathize', 'gen_clarify', 'gen_verify_order',
    'gen_hold', 'gen_transfer', 'gen_apologize', 'gen_close',
    # Presale skills
    'pre_query_product', 'pre_check_stock', 'pre_compare', 'pre_recommend',
    'pre_answer_spec', 'pre_check_promo', 'pre_guide_purchase',
    # Logistics skills
    'log_query_status', 'log_query_detail', 'log_estimate_arrival',
    'log_modify_address', 'log_contact_courier', 'log_delay_notify', 'log_lost_claim',
    # Aftersale skills
    'aft_check_policy', 'aft_collect_evidence', 'aft_initiate_refund',
    'aft_initiate_return', 'aft_initiate_exchange', 'aft_schedule_pickup',
    'aft_track_progress', 'aft_compensate', 'aft_reject_explain'
}


class ValidationError(Exception):
    """Raised when playbook validation fails"""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Validation failed: {'; '.join(errors)}")


def validate_playbook(playbook: dict) -> bool:
    """
    Validate playbook according to design spec Section 6.1.

    Args:
        playbook: The playbook dictionary to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails, with list of error messages
    """
    errors = []

    # 1. Check required fields (including initial_slots)
    required_fields = ['playbook_id', 'session_id', 'scenario', 'subtype', 'initial_slots', 'business_outcome', 'nodes']
    for field in required_fields:
        if field not in playbook:
            errors.append(f"Missing required field: {field}")

    # 1a. Validate business_outcome structure
    if 'business_outcome' in playbook:
        bo = playbook['business_outcome']
        if not isinstance(bo, dict):
            errors.append("business_outcome must be a dict")
        else:
            if 'has_order' not in bo:
                errors.append("business_outcome missing 'has_order' field")
            if 'order_amount' not in bo:
                errors.append("business_outcome missing 'order_amount' field")
            elif not isinstance(bo.get('order_amount'), (int, float)):
                errors.append("business_outcome.order_amount must be a number")

    # 1b. Validate session_id is present and non-empty
    if 'session_id' in playbook:
        if not playbook['session_id']:
            errors.append("session_id cannot be empty")

    # If nodes is missing, we can't continue validation
    if 'nodes' not in playbook:
        raise ValidationError(errors)

    nodes = playbook['nodes']

    # 2. Check skills are valid and not parameterized
    for node_id, node in nodes.items():
        transitions = node.get('transitions', {})
        for skill in transitions.keys():
            # Red Line: Check for parameterized skills
            if '[' in skill or ']' in skill:
                errors.append(f"Skill has parameters: {skill} (node: {node_id})")
            elif skill not in VALID_SKILLS:
                errors.append(f"Invalid skill: {skill} (node: {node_id})")

    # 3. Check each node has default_fallback field
    for node_id, node in nodes.items():
        if 'default_fallback' not in node:
            errors.append(f"Node {node_id} missing default_fallback field")
        else:
            # Check default_fallback references an existing node
            fallback = node['default_fallback']
            if fallback not in nodes:
                errors.append(f"Node {node_id} default_fallback references non-existent node: {fallback}")

    # 4. Check transitions reference valid nodes
    for node_id, node in nodes.items():
        transitions = node.get('transitions', {})
        for skill, target_node in transitions.items():
            if target_node not in nodes:
                errors.append(f"Node {node_id} transition '{skill}' references non-existent node: {target_node}")

    # 5. Check tree structure (each node except terminal must have >= 2 branches)
    for node_id, node in nodes.items():
        transitions = node.get('transitions', {})
        # Terminal nodes and fallback nodes can have empty transitions
        if node_id != 'terminal' and not node_id.startswith('fallback'):
            if len(transitions) < 2:
                errors.append(f"Node {node_id} has insufficient branches ({len(transitions)}), playbook must be a tree not a line!")

    # 6. Check for negative path (at least one node with sentiment='angry')
    has_negative_path = False
    for node in nodes.values():
        if node.get('sentiment') == 'angry':
            has_negative_path = True
            break

    if not has_negative_path:
        errors.append("Playbook missing negative path (angry sentiment node), cannot provide RL punishment signal")

    # 7. Check for unreachable nodes
    all_nodes = set(nodes.keys())
    reachable = {'root'}  # root is always reachable as entry point
    for node in nodes.values():
        reachable.update(node.get('transitions', {}).values())
        fallback = node.get('default_fallback', '')
        if fallback:
            reachable.add(fallback)

    unreachable = all_nodes - reachable - {''}
    if unreachable:
        errors.append(f"Unreachable nodes found: {unreachable}")

    # Raise if any errors
    if errors:
        raise ValidationError(errors)

    return True


def validate_playbook_with_details(playbook: dict) -> Tuple[bool, List[str]]:
    """
    Validate playbook and return (is_valid, errors) tuple.

    This is a non-throwing version for batch validation.

    Args:
        playbook: The playbook dictionary to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    try:
        validate_playbook(playbook)
        return True, []
    except ValidationError as e:
        return False, e.errors