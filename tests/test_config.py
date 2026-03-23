# tests/test_config.py
from etl.config import (
    ROLE_MAPPING, SKILL_DEFINITIONS, STATE_MACHINE,
    VALID_SKILLS, SCENE_CATEGORIES, SCENE_PATTERNS
)

def test_role_mapping_defined():
    """Test role mapping matches design spec"""
    assert ROLE_MAPPING['BUYER'] == 'User'
    assert ROLE_MAPPING['ASSISTANT'] == 'Agent'
    assert ROLE_MAPPING['QA'] == 'Agent'
    assert ROLE_MAPPING['QA_VENDOR'] == 'Agent'
    assert ROLE_MAPPING['MARKETING'] == 'delete'
    assert ROLE_MAPPING['SYSTEM'] == 'slot_update'

def test_skill_definitions_complete():
    """Test all 31 skills defined"""
    assert len(SKILL_DEFINITIONS) == 31
    assert 'gen_greet' in SKILL_DEFINITIONS
    assert 'gen_empathize' in SKILL_DEFINITIONS
    assert 'pre_query_product' in SKILL_DEFINITIONS

def test_state_machine_defined():
    """Test state machine matches design"""
    assert STATE_MACHINE['OPEN'] == ['IDENTIFY']
    assert 'DIAGNOSE' in STATE_MACHINE['IDENTIFY']
    assert 'RESOLVE' in STATE_MACHINE['DIAGNOSE']
    assert 'CLOSE' in STATE_MACHINE['CONFIRM']

def test_valid_skills_set():
    """Test VALID_SKILLS contains all defined skills"""
    assert len(VALID_SKILLS) == 31
    assert VALID_SKILLS == set(SKILL_DEFINITIONS.keys())