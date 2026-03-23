# tests/test_classifier.py
import pytest
from etl.classifier import classify_scene

def test_classify_presale():
    """Test presale scene classification"""
    turns = [
        {'role': 'User', 'text': '我想买这个产品，多少钱？'},
        {'role': 'Agent', 'text': '您好，这个产品价格是99元'},
    ]
    result = classify_scene(turns)
    assert result == 'presale'

def test_classify_logistics():
    """Test logistics scene classification"""
    turns = [
        {'role': 'User', 'text': '我的快递到哪了？'},
        {'role': 'Agent', 'text': '您的快递正在配送中'},
    ]
    result = classify_scene(turns)
    assert result == 'logistics'

def test_classify_aftersale():
    """Test aftersale scene classification"""
    turns = [
        {'role': 'User', 'text': '我要退货，质量有问题'},
        {'role': 'Agent', 'text': '好的，我帮您处理退货'},
    ]
    result = classify_scene(turns)
    assert result == 'aftersale'

def test_classify_unknown():
    """Test unknown scene classification"""
    turns = [
        {'role': 'User', 'text': 'Hello'},
        {'role': 'Agent', 'text': 'Hi'},
    ]
    result = classify_scene(turns)
    assert result == 'unknown'

def test_classify_uses_scene_patterns():
    """Test that classifier uses SCENE_PATTERNS from config"""
    from etl.config import SCENE_PATTERNS
    # Verify the patterns are used
    assert 'presale' in SCENE_PATTERNS
    assert 'logistics' in SCENE_PATTERNS
    assert 'aftersale' in SCENE_PATTERNS


def test_classify_empty_turns():
    """Test empty turns list"""
    result = classify_scene([])
    assert result == 'unknown'


def test_classify_turns_missing_keys():
    """Test turns with missing keys"""
    turns = [
        {'role': 'User'},  # Missing 'text'
        {'text': 'response'},  # Missing 'role'
    ]
    result = classify_scene(turns)
    assert result == 'unknown'


def test_classify_tie_breaking():
    """Test deterministic tie-breaking (presale > logistics > aftersale)"""
    # This has one match each: 购买(presale), 物流(logistics), 退款(aftersale)
    turns = [
        {'role': 'User', 'text': '购买 物流 退款'},  # Ties all 3 categories
    ]
    result = classify_scene(turns)
    assert result == 'presale'  # presale wins ties


def test_classify_only_agent_turns():
    """Test conversation with only Agent turns"""
    turns = [
        {'role': 'Agent', 'text': '你好'},
        {'role': 'Agent', 'text': '有什么可以帮您'},
    ]
    result = classify_scene(turns)
    assert result == 'unknown'