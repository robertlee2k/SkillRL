# tests/test_pipeline.py
"""Tests for the full ETL pipeline."""

import pytest
import json
from etl.pipeline import run_pipeline


def test_run_pipeline_end_to_end(tmp_path):
    """Test complete pipeline from raw data to playbooks"""
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

    # Run pipeline
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

    output_file = tmp_path / 'output.json'
    stats = run_pipeline(str(input_file), str(output_file))

    assert stats['total'] == 2
    assert stats['valid'] == 1
    assert stats['invalid'] == 1


def test_run_pipeline_creates_output_directory(tmp_path):
    """Test pipeline creates output directory if needed"""
    raw_data = [{'session_id': 'S-001', 'messages': [
        {'sent_by': 'BUYER', 'content': 'Hello'},
        {'sent_by': 'ASSISTANT', 'content': 'Hi'},
    ]}]

    input_file = tmp_path / 'input.json'
    input_file.write_text(json.dumps(raw_data, ensure_ascii=False))

    # Use nested path that doesn't exist
    output_file = tmp_path / 'nested' / 'dir' / 'output.json'
    stats = run_pipeline(str(input_file), str(output_file))

    assert output_file.exists()