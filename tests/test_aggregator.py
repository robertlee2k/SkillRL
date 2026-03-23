# tests/test_aggregator.py
"""Tests for turn aggregation core algorithm.

Tests the aggregate_turns function per design spec Section 3.3.
"""

import pytest
from etl.aggregator import aggregate_turns


def test_aggregate_basic_conversation():
    """Test basic User-Agent alternation"""
    messages = [
        {'sent_by': 'BUYER', 'content': 'Hello'},
        {'sent_by': 'ASSISTANT', 'content': 'Hi there'},
        {'sent_by': 'BUYER', 'content': 'I have a question'},
        {'sent_by': 'ASSISTANT', 'content': 'Sure, go ahead'},
    ]
    result = aggregate_turns(messages)
    assert len(result['turns']) == 4
    assert result['turns'][0]['role'] == 'User'
    assert result['turns'][1]['role'] == 'Agent'
    assert result['initial_slots'] == {}


def test_aggregate_consecutive_same_role():
    """Test aggregation of consecutive messages from same role"""
    messages = [
        {'sent_by': 'BUYER', 'content': 'Hello'},
        {'sent_by': 'BUYER', 'content': 'Are you there?'},
        {'sent_by': 'ASSISTANT', 'content': 'Yes'},
    ]
    result = aggregate_turns(messages)
    assert len(result['turns']) == 2
    assert 'Hello' in result['turns'][0]['text']
    assert 'Are you there?' in result['turns'][0]['text']


def test_aggregate_marketing_deleted():
    """Test MARKETING messages are deleted"""
    messages = [
        {'sent_by': 'BUYER', 'content': 'Hello'},
        {'sent_by': 'MARKETING', 'content': 'Buy now!'},
        {'sent_by': 'ASSISTANT', 'content': 'Hi'},
    ]
    result = aggregate_turns(messages)
    assert len(result['turns']) == 2
    marketing_contents = [t['text'] for t in result['turns']]
    assert 'Buy now!' not in marketing_contents


def test_aggregate_system_slots():
    """Test SYSTEM messages update slots"""
    messages = [
        {'sent_by': 'SYSTEM', 'content': '买家已付款'},
        {'sent_by': 'BUYER', 'content': 'Hello'},
        {'sent_by': 'ASSISTANT', 'content': 'Hi'},
    ]
    result = aggregate_turns(messages)
    assert result['initial_slots'] == {'order_paid': True}


def test_aggregate_system_mid_conversation():
    """Test SYSTEM messages mid-conversation attach to Agent"""
    messages = [
        {'sent_by': 'BUYER', 'content': 'Hello'},
        {'sent_by': 'ASSISTANT', 'content': 'Hi'},
        {'sent_by': 'SYSTEM', 'content': '买家已付款'},
        {'sent_by': 'BUYER', 'content': 'Thanks'},
    ]
    result = aggregate_turns(messages)
    assert result['turns'][1]['slot_updates'] == {'order_paid': True}


def test_initial_slots_edge_case():
    """Test SYSTEM at start goes to initial_slots (Red Line Fix)"""
    messages = [
        {'sent_by': 'SYSTEM', 'content': '买家已付款'},
        {'sent_by': 'BUYER', 'content': 'Hello'},
    ]
    result = aggregate_turns(messages)
    assert result['initial_slots'] == {'order_paid': True}
    assert len(result['turns']) == 1


def test_aggregate_qa_as_agent():
    """Test QA messages are mapped to Agent role"""
    messages = [
        {'sent_by': 'BUYER', 'content': 'Hello'},
        {'sent_by': 'QA', 'content': 'Hi there'},
    ]
    result = aggregate_turns(messages)
    assert len(result['turns']) == 2
    assert result['turns'][1]['role'] == 'Agent'


def test_aggregate_qa_vendor_as_agent():
    """Test QA_VENDOR messages are mapped to Agent role"""
    messages = [
        {'sent_by': 'BUYER', 'content': 'Hello'},
        {'sent_by': 'QA_VENDOR', 'content': 'Hi there'},
    ]
    result = aggregate_turns(messages)
    assert len(result['turns']) == 2
    assert result['turns'][1]['role'] == 'Agent'


def test_aggregate_consecutive_agents():
    """Test consecutive ASSISTANT/QA messages are aggregated"""
    messages = [
        {'sent_by': 'BUYER', 'content': 'Hello'},
        {'sent_by': 'QA', 'content': 'Quick answer'},
        {'sent_by': 'ASSISTANT', 'content': 'Let me clarify'},
    ]
    result = aggregate_turns(messages)
    assert len(result['turns']) == 2
    assert 'Quick answer' in result['turns'][1]['text']
    assert 'Let me clarify' in result['turns'][1]['text']


def test_system_after_user_goes_to_initial_slots():
    """Test SYSTEM after User message goes to initial_slots (edge case)"""
    messages = [
        {'sent_by': 'BUYER', 'content': 'Hello'},
        {'sent_by': 'SYSTEM', 'content': '买家已付款'},
        {'sent_by': 'ASSISTANT', 'content': 'Hi'},
    ]
    result = aggregate_turns(messages)
    # SYSTEM after User goes to initial_slots, not Agent's slot_updates
    assert result['initial_slots'] == {'order_paid': True}
    assert result['turns'][1]['slot_updates'] == {}


def test_system_ignored_message():
    """Test SYSTEM messages that should be ignored (无意义提示)"""
    messages = [
        {'sent_by': 'SYSTEM', 'content': '买家已读'},
        {'sent_by': 'BUYER', 'content': 'Hello'},
    ]
    result = aggregate_turns(messages)
    assert result['initial_slots'] == {}
    assert len(result['turns']) == 1


def test_multiple_system_at_start():
    """Test multiple SYSTEM messages at start accumulate in initial_slots"""
    messages = [
        {'sent_by': 'SYSTEM', 'content': '买家已付款'},
        {'sent_by': 'SYSTEM', 'content': '订单已签收'},
        {'sent_by': 'BUYER', 'content': 'Hello'},
    ]
    result = aggregate_turns(messages)
    assert result['initial_slots'] == {'order_paid': True, 'order_signed': True}