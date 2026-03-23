# tests/test_parser.py
import pytest
from etl.parser import parse_system_message, extract_slot_updates, SYSTEM_SLOT_PATTERNS


class TestSystemSlotPatterns:
    """Test SYSTEM_SLOT_PATTERNS constant matches design spec."""

    def test_patterns_exist(self):
        """Test that required patterns are defined."""
        assert "买家已发起退款" in SYSTEM_SLOT_PATTERNS
        assert "订单已签收" in SYSTEM_SLOT_PATTERNS
        assert "买家已付款" in SYSTEM_SLOT_PATTERNS
        assert "买家已读" in SYSTEM_SLOT_PATTERNS
        assert "消息已送达" in SYSTEM_SLOT_PATTERNS

    def test_refund_initiated_pattern(self):
        """Test refund_initiated slot pattern."""
        assert SYSTEM_SLOT_PATTERNS["买家已发起退款"] == {"refund_initiated": True}

    def test_order_signed_pattern(self):
        """Test order_signed slot pattern."""
        assert SYSTEM_SLOT_PATTERNS["订单已签收"] == {"order_signed": True}

    def test_order_paid_pattern(self):
        """Test order_paid slot pattern."""
        assert SYSTEM_SLOT_PATTERNS["买家已付款"] == {"order_paid": True}

    def test_drop_patterns_are_none(self):
        """Test that drop patterns return None."""
        assert SYSTEM_SLOT_PATTERNS["买家已读"] is None
        assert SYSTEM_SLOT_PATTERNS["消息已送达"] is None


class TestParseSystemMessage:
    """Test parse_system_message function."""

    def test_parse_system_message_with_slots(self):
        """Test parsing SYSTEM message with slot updates."""
        msg = {
            'sent_by': 'SYSTEM',
            'content': '{"slot_update": {"order_id_collected": true, "product_id_collected": true}}'
        }
        result = parse_system_message(msg)
        assert result == {'order_id_collected': True, 'product_id_collected': True}

    def test_parse_system_message_no_slots(self):
        """Test SYSTEM message without slot updates."""
        msg = {
            'sent_by': 'SYSTEM',
            'content': 'System notification message'
        }
        result = parse_system_message(msg)
        assert result == {}

    def test_parse_system_message_empty(self):
        """Test empty SYSTEM message."""
        msg = {'sent_by': 'SYSTEM', 'content': ''}
        result = parse_system_message(msg)
        assert result == {}

    def test_parse_system_message_refund_initiated(self):
        """Test Chinese pattern: 买家已发起退款."""
        msg = {'sent_by': 'SYSTEM', 'content': '买家已发起退款'}
        result = parse_system_message(msg)
        assert result == {'refund_initiated': True}

    def test_parse_system_message_order_signed(self):
        """Test Chinese pattern: 订单已签收."""
        msg = {'sent_by': 'SYSTEM', 'content': '订单已签收'}
        result = parse_system_message(msg)
        assert result == {'order_signed': True}

    def test_parse_system_message_order_paid(self):
        """Test Chinese pattern: 买家已付款."""
        msg = {'sent_by': 'SYSTEM', 'content': '买家已付款'}
        result = parse_system_message(msg)
        assert result == {'order_paid': True}

    def test_parse_system_message_drop_read(self):
        """Test that 买家已读 returns empty (drop)."""
        msg = {'sent_by': 'SYSTEM', 'content': '买家已读'}
        result = parse_system_message(msg)
        assert result == {}

    def test_parse_system_message_drop_delivered(self):
        """Test that 消息已送达 returns empty (drop)."""
        msg = {'sent_by': 'SYSTEM', 'content': '消息已送达'}
        result = parse_system_message(msg)
        assert result == {}


class TestExtractSlotUpdates:
    """Test extract_slot_updates function."""

    def test_extract_order_id_sets_boolean_slot(self):
        """Test that finding order ID sets order_id_collected to True."""
        text = "您的订单号是 ORD-12345"
        result = extract_slot_updates(text)
        assert result == {'order_id_collected': True}

    def test_extract_product_id_sets_boolean_slot(self):
        """Test that finding product ID sets product_id_collected to True."""
        text = "商品ID为 PROD-67890"
        result = extract_slot_updates(text)
        assert result == {'product_id_collected': True}

    def test_extract_both_ids(self):
        """Test extracting both order and product IDs."""
        text = "您的订单号是 ORD-12345，商品ID为 PROD-67890"
        result = extract_slot_updates(text)
        assert result == {'order_id_collected': True, 'product_id_collected': True}

    def test_extract_no_ids(self):
        """Test text without order or product IDs."""
        text = "这是一条普通消息"
        result = extract_slot_updates(text)
        assert result == {}

    def test_no_phone_slot(self):
        """Test that phone numbers are NOT extracted (removed per spec)."""
        text = "联系我: 13812345678"
        result = extract_slot_updates(text)
        assert 'phone' not in result
        assert result == {}

    def test_case_insensitive_order_id(self):
        """Test case insensitive order ID matching."""
        text = "订单: ord-99999"
        result = extract_slot_updates(text)
        assert result == {'order_id_collected': True}

    def test_case_insensitive_product_id(self):
        """Test case insensitive product ID matching."""
        text = "商品: prod-11111"
        result = extract_slot_updates(text)
        assert result == {'product_id_collected': True}