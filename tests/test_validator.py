# tests/test_validator.py
"""Test playbook validation per design spec Section 6.1"""
import pytest
from etl.validator import validate_playbook, ValidationError


def test_validate_valid_playbook():
    """Test validation passes for valid playbook"""
    playbook = {
        'playbook_id': 'logistics_001_delay',
        'scenario': 'logistics',
        'subtype': 'logistics_query',
        'initial_slots': {},
        'nodes': {
            'root': {
                'buyer_text': 'Hello',
                'sentiment': 'calm',
                'slot_updates': {},
                'transitions': {'gen_greet': 'node_1', 'gen_empathize': 'node_2'},
                'default_fallback': 'fallback'
            },
            'node_1': {
                'buyer_text': 'Hi',
                'sentiment': 'angry',
                'slot_updates': {},
                'transitions': {'gen_close': 'terminal', 'gen_apologize': 'node_2'},
                'default_fallback': 'terminal'
            },
            'node_2': {
                'buyer_text': 'Thanks',
                'sentiment': 'calm',
                'slot_updates': {},
                'transitions': {'gen_close': 'terminal', 'gen_greet': 'terminal'},
                'default_fallback': 'terminal'
            },
            'terminal': {
                'buyer_text': '[END]',
                'sentiment': 'calm',
                'slot_updates': {},
                'transitions': {},
                'default_fallback': 'terminal'
            },
            'fallback': {
                'buyer_text': 'Confused',
                'sentiment': 'neutral',
                'slot_updates': {},
                'transitions': {'gen_clarify': 'root', 'gen_close': 'terminal'},
                'default_fallback': 'terminal'
            }
        }
    }
    result = validate_playbook(playbook)
    assert result == True


def test_validate_missing_required_field():
    """Test validation fails for missing required field"""
    playbook = {
        'playbook_id': 'test_001',
        # missing 'nodes', 'scenario', etc.
    }
    with pytest.raises(ValidationError):
        validate_playbook(playbook)


def test_validate_invalid_skill():
    """Test validation fails for invalid skill"""
    playbook = {
        'playbook_id': 'test_001',
        'scenario': 'presale',
        'subtype': 'test',
        'initial_slots': {},
        'nodes': {
            'root': {
                'buyer_text': 'Hello',
                'sentiment': 'calm',
                'transitions': {'invalid_skill': 'terminal'},
                'default_fallback': 'terminal'
            },
            'terminal': {
                'buyer_text': '[END]',
                'sentiment': 'calm',
                'transitions': {},
                'default_fallback': 'terminal'
            }
        }
    }
    with pytest.raises(ValidationError):
        validate_playbook(playbook)


def test_validate_no_parameterized_skills():
    """Test that parameterized skills are rejected (Red Line)"""
    playbook = {
        'playbook_id': 'test_001',
        'scenario': 'presale',
        'subtype': 'test',
        'initial_slots': {},
        'nodes': {
            'root': {
                'buyer_text': 'Hello',
                'sentiment': 'calm',
                'transitions': {'query_product[order_id=123]': 'terminal'},
                'default_fallback': 'terminal'
            },
            'terminal': {
                'buyer_text': '[END]',
                'sentiment': 'calm',
                'transitions': {},
                'default_fallback': 'terminal'
            }
        }
    }
    with pytest.raises(ValidationError):
        validate_playbook(playbook)


def test_validate_missing_default_fallback():
    """Test that missing default_fallback raises error"""
    playbook = {
        'playbook_id': 'test_001',
        'scenario': 'presale',
        'subtype': 'test',
        'initial_slots': {},
        'nodes': {
            'root': {
                'buyer_text': 'Hello',
                'sentiment': 'calm',
                'transitions': {'gen_greet': 'terminal'},
                # Missing default_fallback!
            },
            'terminal': {
                'buyer_text': '[END]',
                'sentiment': 'calm',
                'transitions': {},
                'default_fallback': 'terminal'
            }
        }
    }
    with pytest.raises(ValidationError):
        validate_playbook(playbook)


def test_validate_linear_structure():
    """Test that linear structure (single branch) is rejected"""
    playbook = {
        'playbook_id': 'test_001',
        'scenario': 'presale',
        'subtype': 'test',
        'initial_slots': {},
        'nodes': {
            'root': {
                'buyer_text': 'Hello',
                'sentiment': 'calm',  # No angry node!
                'transitions': {'gen_greet': 'node_1'},  # Only 1 branch
                'default_fallback': 'terminal'
            },
            'node_1': {
                'buyer_text': 'Hi',
                'sentiment': 'calm',
                'transitions': {'gen_close': 'terminal'},  # Only 1 branch
                'default_fallback': 'terminal'
            },
            'terminal': {
                'buyer_text': '[END]',
                'sentiment': 'calm',
                'transitions': {},
                'default_fallback': 'terminal'
            }
        }
    }
    with pytest.raises(ValidationError):
        validate_playbook(playbook)


def test_validate_no_negative_path():
    """Test that missing angry sentiment (no negative path) raises error"""
    playbook = {
        'playbook_id': 'test_001',
        'scenario': 'presale',
        'subtype': 'test',
        'initial_slots': {},
        'nodes': {
            'root': {
                'buyer_text': 'Hello',
                'sentiment': 'calm',  # No angry node!
                'transitions': {'gen_greet': 'node_1', 'gen_empathize': 'node_2'},
                'default_fallback': 'terminal'
            },
            'node_1': {
                'buyer_text': 'Hi',
                'sentiment': 'calm',
                'transitions': {'gen_close': 'terminal'},
                'default_fallback': 'terminal'
            },
            'node_2': {
                'buyer_text': 'Thanks',
                'sentiment': 'neutral',
                'transitions': {'gen_close': 'terminal'},
                'default_fallback': 'terminal'
            },
            'terminal': {
                'buyer_text': '[END]',
                'sentiment': 'calm',
                'transitions': {},
                'default_fallback': 'terminal'
            }
        }
    }
    with pytest.raises(ValidationError):
        validate_playbook(playbook)


def test_validate_invalid_fallback_reference():
    """Test that default_fallback referencing non-existent node raises error"""
    playbook = {
        'playbook_id': 'test_001',
        'scenario': 'presale',
        'subtype': 'test',
        'initial_slots': {},
        'nodes': {
            'root': {
                'buyer_text': 'Hello',
                'sentiment': 'angry',
                'transitions': {'gen_greet': 'node_1', 'gen_empathize': 'terminal'},
                'default_fallback': 'nonexistent_node'  # References non-existent node
            },
            'node_1': {
                'buyer_text': 'Hi',
                'sentiment': 'calm',
                'transitions': {'gen_close': 'terminal'},
                'default_fallback': 'terminal'
            },
            'terminal': {
                'buyer_text': '[END]',
                'sentiment': 'calm',
                'transitions': {},
                'default_fallback': 'terminal'
            }
        }
    }
    with pytest.raises(ValidationError):
        validate_playbook(playbook)


def test_validate_transition_to_nonexistent_node():
    """Test that transition to non-existent node raises error"""
    playbook = {
        'playbook_id': 'test_001',
        'scenario': 'presale',
        'subtype': 'test',
        'initial_slots': {},
        'nodes': {
            'root': {
                'buyer_text': 'Hello',
                'sentiment': 'angry',
                'transitions': {'gen_greet': 'nonexistent_node', 'gen_empathize': 'terminal'},
                'default_fallback': 'terminal'
            },
            'terminal': {
                'buyer_text': '[END]',
                'sentiment': 'calm',
                'transitions': {},
                'default_fallback': 'terminal'
            }
        }
    }
    with pytest.raises(ValidationError):
        validate_playbook(playbook)


def test_validate_unreachable_nodes():
    """Test that unreachable nodes raise error"""
    playbook = {
        'playbook_id': 'test_001',
        'scenario': 'presale',
        'subtype': 'test',
        'initial_slots': {},
        'nodes': {
            'root': {
                'buyer_text': 'Hello',
                'sentiment': 'angry',
                'transitions': {'gen_greet': 'node_1', 'gen_empathize': 'terminal'},
                'default_fallback': 'terminal'
            },
            'node_1': {
                'buyer_text': 'Hi',
                'sentiment': 'calm',
                'transitions': {'gen_close': 'terminal'},
                'default_fallback': 'terminal'
            },
            'terminal': {
                'buyer_text': '[END]',
                'sentiment': 'calm',
                'transitions': {},
                'default_fallback': 'terminal'
            },
            'orphan_node': {  # This node is never referenced by any transition
                'buyer_text': 'Orphan',
                'sentiment': 'calm',
                'transitions': {'gen_close': 'terminal'},
                'default_fallback': 'terminal'
            }
        }
    }
    with pytest.raises(ValidationError):
        validate_playbook(playbook)