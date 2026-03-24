# tests/test_pipeline.py
"""Tests for the full ETL pipeline."""

import json
from unittest.mock import patch


def test_run_pipeline_end_to_end(tmp_path):
    """Test complete pipeline from raw data to playbooks with mocked LLM"""
    # Create test input
    raw_data = [
        {
            'session_id': 'S-001',
            'messages': [
                {'sent_by': 'BUYER', 'content': '我想买这个产品，多少钱？'},
                {'sent_by': 'ASSISTANT', 'content': '您好，这个产品价格是99元'},
                {'sent_by': 'BUYER', 'content': '好的，我买一个'},
                {'sent_by': 'ASSISTANT', 'content': '好的，已为您下单'},
            ]
        }
    ]
    input_file = tmp_path / 'input.json'
    input_file.write_text(json.dumps(raw_data, ensure_ascii=False))

    # Mock LLM responses - valid playbook with tree structure and negative path
    mock_playbook = {
        'nodes': {
            'root': {
                'buyer_text': '我想买这个产品，多少钱？',
                'sentiment': 'calm',
                'slot_updates': {},
                'available_skills': ['gen_greet', 'pre_query_product', 'gen_clarify'],
                'transitions': {
                    'gen_greet': 'node_1',
                    'pre_query_product': 'node_1',
                    'gen_clarify': 'fallback'
                },
                'default_fallback': 'fallback'
            },
            'node_1': {
                'buyer_text': '好的，我买一个',
                'sentiment': 'angry',  # Negative path for RL punishment
                'slot_updates': {},
                'available_skills': ['pre_guide_purchase', 'gen_close', 'gen_empathize'],
                'transitions': {
                    'pre_guide_purchase': 'node_2',
                    'gen_close': 'terminal',
                    'gen_empathize': 'node_2'
                },
                'default_fallback': 'fallback'
            },
            'node_2': {
                'buyer_text': '谢谢',
                'sentiment': 'calm',
                'slot_updates': {},
                'available_skills': ['gen_close', 'gen_empathize'],
                'transitions': {
                    'gen_close': 'terminal',
                    'gen_empathize': 'terminal'
                },
                'default_fallback': 'terminal'
            },
            'fallback': {
                'buyer_text': '不太理解您的意思',
                'sentiment': 'neutral',
                'slot_updates': {},
                'available_skills': ['gen_clarify', 'gen_transfer'],
                'transitions': {
                    'gen_clarify': 'node_1',
                    'gen_transfer': 'terminal'
                },
                'default_fallback': 'terminal'
            },
            'terminal': {
                'buyer_text': '[END]',
                'sentiment': 'calm',
                'slot_updates': {},
                'available_skills': [],
                'transitions': {},
                'default_fallback': 'terminal'
            }
        }
    }

    with patch('etl.llm_generator.call_llm_for_classification') as mock_classify, \
         patch('etl.llm_generator.call_llm_for_playbook') as mock_playbook_llm:
        mock_classify.return_value = 'presale'
        mock_playbook_llm.return_value = mock_playbook

        from etl.pipeline import run_pipeline
        output_file = tmp_path / 'output.json'
        stats = run_pipeline(str(input_file), str(output_file))

        # Check stats
        assert stats['total'] == 1
        assert stats['valid'] == 1
        assert stats['invalid'] == 0

        # Check output
        output = json.loads(output_file.read_text())
        assert len(output) == 1
        assert output[0]['session_id'] == 'S-001'
        assert output[0]['scenario'] == 'presale'


def test_run_pipeline_with_invalid_session(tmp_path):
    """Test pipeline filters invalid sessions"""
    raw_data = [
        {
            'session_id': 'S-001',
            'messages': [
                {'sent_by': 'BUYER', 'content': 'Valid'},
                {'sent_by': 'ASSISTANT', 'content': 'Response'},
            ]
        },
        {
            'session_id': 'S-002',
            'messages': [
                {'sent_by': 'BUYER', 'content': 'Only one message'},
            ]
        }
    ]
    input_file = tmp_path / 'input.json'
    input_file.write_text(json.dumps(raw_data, ensure_ascii=False))

    with patch('etl.llm_generator.call_llm_for_classification') as mock_classify, \
         patch('etl.llm_generator.call_llm_for_playbook') as mock_playbook_llm:
        mock_classify.return_value = 'unknown'
        # Valid playbook with angry sentiment and tree structure
        mock_playbook_llm.return_value = {
            'nodes': {
                'root': {
                    'buyer_text': 'Valid',
                    'sentiment': 'angry',  # Negative path
                    'slot_updates': {},
                    'available_skills': ['gen_greet', 'gen_clarify'],
                    'transitions': {'gen_greet': 'terminal', 'gen_clarify': 'terminal'},
                    'default_fallback': 'terminal'
                },
                'terminal': {
                    'buyer_text': '[END]',
                    'sentiment': 'calm',
                    'slot_updates': {},
                    'available_skills': [],
                    'transitions': {},
                    'default_fallback': 'terminal'
                }
            }
        }

        from etl.pipeline import run_pipeline
        output_file = tmp_path / 'output.json'
        stats = run_pipeline(str(input_file), str(output_file))

        assert stats['total'] == 2
        assert stats['valid'] == 1
        assert stats['invalid'] == 1  # S-002 has only 1 turn, fails min_turns


def test_run_pipeline_creates_output_directory(tmp_path):
    """Test pipeline creates output directory if needed"""
    raw_data = [{'session_id': 'S-001', 'messages': [
        {'sent_by': 'BUYER', 'content': 'Hello'},
        {'sent_by': 'ASSISTANT', 'content': 'Hi'},
    ]}]

    input_file = tmp_path / 'input.json'
    input_file.write_text(json.dumps(raw_data, ensure_ascii=False))

    with patch('etl.llm_generator.call_llm_for_classification') as mock_classify, \
         patch('etl.llm_generator.call_llm_for_playbook') as mock_playbook_llm:
        mock_classify.return_value = 'unknown'
        # Valid playbook with angry sentiment
        mock_playbook_llm.return_value = {
            'nodes': {
                'root': {
                    'buyer_text': 'Hello',
                    'sentiment': 'angry',  # Negative path
                    'slot_updates': {},
                    'available_skills': ['gen_greet', 'gen_clarify'],
                    'transitions': {'gen_greet': 'terminal', 'gen_clarify': 'terminal'},
                    'default_fallback': 'terminal'
                },
                'terminal': {
                    'buyer_text': '[END]',
                    'sentiment': 'calm',
                    'slot_updates': {},
                    'available_skills': [],
                    'transitions': {},
                    'default_fallback': 'terminal'
                }
            }
        }

        from etl.pipeline import run_pipeline
        # Use nested path that doesn't exist
        output_file = tmp_path / 'nested' / 'dir' / 'output.json'
        run_pipeline(str(input_file), str(output_file))

        assert output_file.exists()


def test_run_pipeline_llm_failure(tmp_path):
    """Test that LLM failure results in invalid session"""
    raw_data = [{'session_id': 'S-001', 'messages': [
        {'sent_by': 'BUYER', 'content': 'Hello'},
        {'sent_by': 'ASSISTANT', 'content': 'Hi'},
    ]}]

    input_file = tmp_path / 'input.json'
    input_file.write_text(json.dumps(raw_data, ensure_ascii=False))

    with patch('etl.llm_generator.call_llm_for_classification') as mock_classify, \
         patch('etl.llm_generator.call_llm_for_playbook') as mock_playbook_llm:
        mock_classify.return_value = 'unknown'
        mock_playbook_llm.return_value = None  # LLM failed

        from etl.pipeline import run_pipeline
        output_file = tmp_path / 'output.json'
        stats = run_pipeline(str(input_file), str(output_file))

        assert stats['valid'] == 0
        assert stats['invalid'] >= 1  # At least the LLM failure