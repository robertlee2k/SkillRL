# tests/test_batch.py
import pytest
import json
from etl.batch import process_batch, load_sessions, save_playbooks

def test_process_batch(tmp_path):
    """Test batch processing of sessions"""
    sessions = [
        {
            'session_id': 'S-001',
            'messages': [
                {'sent_by': 'BUYER', 'content': 'Hello'},
                {'sent_by': 'ASSISTANT', 'content': 'Hi'},
            ]
        },
        {
            'session_id': 'S-002',
            'messages': [
                {'sent_by': 'BUYER', 'content': 'Question'},
                {'sent_by': 'ASSISTANT', 'content': 'Answer'},
            ]
        }
    ]
    result = process_batch(sessions)
    assert len(result['playbooks']) == 2
    assert result['stats']['total'] == 2
    assert result['stats']['valid'] == 2
    assert result['stats']['invalid'] == 0

def test_process_batch_filters_invalid(tmp_path):
    """Test batch processing filters out invalid sessions"""
    sessions = [
        {
            'session_id': 'S-001',
            'messages': [
                {'sent_by': 'BUYER', 'content': 'Hello'},
                {'sent_by': 'ASSISTANT', 'content': 'Hi'},
            ]
        },
        {
            'session_id': 'S-002',
            'messages': [
                {'sent_by': 'BUYER', 'content': 'Only one message'},
            ]
        }
    ]
    result = process_batch(sessions, min_turns=2)
    assert len(result['playbooks']) == 1
    assert result['playbooks'][0]['session_id'] == 'S-001'
    assert result['stats']['total'] == 2
    assert result['stats']['valid'] == 1
    assert result['stats']['invalid'] == 1

def test_load_sessions(tmp_path):
    """Test loading sessions from JSON file"""
    data = [
        {'session_id': 'S-001', 'messages': []},
        {'session_id': 'S-002', 'messages': []},
    ]
    input_file = tmp_path / 'input.json'
    input_file.write_text(json.dumps(data))

    sessions = load_sessions(str(input_file))
    assert len(sessions) == 2

def test_load_sessions_file_not_found(tmp_path):
    """Test loading sessions from non-existent file raises FileNotFoundError"""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_sessions(str(tmp_path / 'nonexistent.json'))
    assert "Session file not found" in str(exc_info.value)

def test_load_sessions_invalid_json(tmp_path):
    """Test loading sessions from invalid JSON raises ValueError"""
    input_file = tmp_path / 'invalid.json'
    input_file.write_text("not valid json")

    with pytest.raises(ValueError) as exc_info:
        load_sessions(str(input_file))
    assert "Invalid JSON" in str(exc_info.value)

def test_save_playbooks(tmp_path):
    """Test saving playbooks to JSON file"""
    playbooks = [
        {'session_id': 'S-001', 'turns': []},
        {'session_id': 'S-002', 'turns': []},
    ]
    output_file = tmp_path / 'output.json'
    save_playbooks(playbooks, str(output_file))

    loaded = json.loads(output_file.read_text())
    assert len(loaded) == 2