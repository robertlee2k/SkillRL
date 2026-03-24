# tests/test_classifier.py
"""Tests for scene classification using LLM."""

from unittest.mock import patch


def test_classify_presale():
    """Test presale scene classification via LLM"""
    turns = [
        {'role': 'User', 'text': '我想买这个产品，多少钱？'},
        {'role': 'Agent', 'text': '您好，这个产品价格是99元'},
    ]
    with patch('etl.llm_generator.call_llm_for_classification') as mock_llm:
        mock_llm.return_value = 'presale'
        from etl.classifier import classify_scene
        result = classify_scene(turns)
        assert result == 'presale'


def test_classify_logistics():
    """Test logistics scene classification via LLM"""
    turns = [
        {'role': 'User', 'text': '我的快递到哪了？'},
        {'role': 'Agent', 'text': '您的快递正在配送中'},
    ]
    with patch('etl.llm_generator.call_llm_for_classification') as mock_llm:
        mock_llm.return_value = 'logistics'
        from etl.classifier import classify_scene
        result = classify_scene(turns)
        assert result == 'logistics'


def test_classify_aftersale():
    """Test aftersale scene classification via LLM"""
    turns = [
        {'role': 'User', 'text': '我要退货，质量有问题'},
        {'role': 'Agent', 'text': '好的，我帮您处理退货'},
    ]
    with patch('etl.llm_generator.call_llm_for_classification') as mock_llm:
        mock_llm.return_value = 'aftersale'
        from etl.classifier import classify_scene
        result = classify_scene(turns)
        assert result == 'aftersale'


def test_classify_trash_mapped_to_unknown():
    """Test that 'trash' from LLM is mapped to 'unknown'"""
    turns = [
        {'role': 'User', 'text': 'Hello'},
        {'role': 'Agent', 'text': 'Hi'},
    ]
    with patch('etl.llm_generator.call_llm_for_classification') as mock_llm:
        mock_llm.return_value = 'trash'
        from etl.classifier import classify_scene
        result = classify_scene(turns)
        assert result == 'unknown'


def test_classify_empty_turns():
    """Test empty turns list returns unknown"""
    from etl.classifier import classify_scene
    result = classify_scene([])
    assert result == 'unknown'


def test_classify_turns_missing_keys():
    """Test turns with missing keys returns unknown"""
    turns = [
        {'role': 'User'},  # Missing 'text'
        {'text': 'response'},  # Missing 'role'
    ]
    from etl.classifier import classify_scene
    result = classify_scene(turns)
    assert result == 'unknown'


def test_classify_only_agent_turns():
    """Test conversation with only Agent turns returns unknown"""
    turns = [
        {'role': 'Agent', 'text': '你好'},
        {'role': 'Agent', 'text': '有什么可以帮您'},
    ]
    from etl.classifier import classify_scene
    result = classify_scene(turns)
    assert result == 'unknown'


def test_classify_llm_failure_returns_unknown():
    """Test that LLM failure returns unknown"""
    turns = [
        {'role': 'User', 'text': '测试对话'},
    ]
    with patch('etl.llm_generator.call_llm_for_classification') as mock_llm:
        mock_llm.side_effect = Exception("LLM error")
        from etl.classifier import classify_scene
        result = classify_scene(turns)
        assert result == 'unknown'