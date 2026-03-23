# tests/test_cleaner.py
import pytest
from etl.cleaner import clean_session, validate_user_agent_alternation


def test_clean_session_basic():
    """Test basic session cleaning"""
    session = {
        'session_id': 'S-001',
        'messages': [
            {'sent_by': 'BUYER', 'content': 'Hello'},
            {'sent_by': 'ASSISTANT', 'content': 'Hi'},
        ]
    }
    result = clean_session(session)
    assert 'turns' in result
    assert len(result['turns']) == 2


def test_validate_user_agent_alternation_valid():
    """Test valid User-Agent alternation"""
    turns = [
        {'role': 'User', 'text': 'Hello'},
        {'role': 'Agent', 'text': 'Hi'},
        {'role': 'User', 'text': 'Thanks'},
    ]
    assert validate_user_agent_alternation(turns) == True


def test_validate_user_agent_alternation_invalid():
    """Test invalid alternation (consecutive same roles)"""
    turns = [
        {'role': 'User', 'text': 'Hello'},
        {'role': 'User', 'text': 'Are you there?'},
        {'role': 'Agent', 'text': 'Hi'},
    ]
    assert validate_user_agent_alternation(turns) == False


def test_clean_session_min_length():
    """Test session with too few turns is rejected"""
    session = {
        'session_id': 'S-001',
        'messages': [
            {'sent_by': 'BUYER', 'content': 'Hello'},
        ]
    }
    result = clean_session(session, min_turns=2)
    assert result is None


def test_clean_session_starts_with_user():
    """Test session must start with User"""
    session = {
        'session_id': 'S-001',
        'messages': [
            {'sent_by': 'ASSISTANT', 'content': 'Hello'},
            {'sent_by': 'BUYER', 'content': 'Hi'},
        ]
    }
    result = clean_session(session)
    assert result is None or result['turns'][0]['role'] == 'User'


def test_clean_session_with_initial_slots():
    """Test initial_slots is passed through"""
    session = {
        'session_id': 'S-001',
        'messages': [
            {'sent_by': 'SYSTEM', 'content': '买家已付款'},
            {'sent_by': 'BUYER', 'content': 'Hello'},
            {'sent_by': 'ASSISTANT', 'content': 'Hi'},
        ]
    }
    result = clean_session(session)
    assert result['initial_slots'] == {'order_paid': True}