import pytest
import json
from pathlib import Path

# Project root for finding sample data
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def test_integration_with_sample_data(tmp_path):
    """Test with actual sample data structure"""
    # Use relative path from project root
    sample_path = PROJECT_ROOT / 'data_sample_100.json'
    try:
        with open(sample_path) as f:
            sample_data = json.load(f)
    except FileNotFoundError:
        pytest.skip("Sample data not found")

    from etl import run_pipeline

    # Handle both dict and list formats
    if isinstance(sample_data, dict):
        sample_data = sample_data.get('sessions', [])

    # Take first 5 sessions
    test_data = sample_data[:5]
    input_file = tmp_path / 'test_input.json'
    input_file.write_text(json.dumps(test_data, ensure_ascii=False))

    # Run pipeline
    output_file = tmp_path / 'test_output.json'
    stats = run_pipeline(str(input_file), str(output_file))

    # Verify stats
    assert stats['total'] == 5
    assert 'valid' in stats
    assert 'invalid' in stats

    # Verify output file exists
    assert output_file.exists()

    # Verify output structure
    output = json.loads(output_file.read_text())
    for playbook in output:
        assert 'playbook_id' in playbook
        assert 'scenario' in playbook
        assert 'nodes' in playbook
        assert 'initial_slots' in playbook


def test_all_exports_available():
    """Test that all expected exports are available"""
    from etl import (
        # Config
        ROLE_MAPPING, SKILL_DEFINITIONS, STATE_MACHINE,
        VALID_SKILLS, SCENE_CATEGORIES, SCENE_PATTERNS,
        # Parser
        parse_system_message, extract_slot_updates, SYSTEM_SLOT_PATTERNS,
        # Aggregator
        aggregate_turns,
        # Cleaner
        clean_session, validate_user_agent_alternation,
        # Batch
        load_sessions, save_playbooks, process_batch,
        # Classifier
        classify_scene,
        # Validator
        validate_playbook, ValidationError, validate_playbook_with_details,
        # Pipeline
        run_pipeline, build_playbook
    )
    # Verify all are not None
    assert ROLE_MAPPING is not None
    assert SKILL_DEFINITIONS is not None
    assert STATE_MACHINE is not None
    assert VALID_SKILLS is not None
    assert SCENE_CATEGORIES is not None
    assert SCENE_PATTERNS is not None
    assert parse_system_message is not None
    assert extract_slot_updates is not None
    assert SYSTEM_SLOT_PATTERNS is not None
    assert aggregate_turns is not None
    assert clean_session is not None
    assert validate_user_agent_alternation is not None
    assert load_sessions is not None
    assert save_playbooks is not None
    assert process_batch is not None
    assert classify_scene is not None
    assert validate_playbook is not None
    assert ValidationError is not None
    assert validate_playbook_with_details is not None
    assert run_pipeline is not None
    assert build_playbook is not None