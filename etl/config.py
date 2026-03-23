# etl/config.py
# Role mapping from raw data to Playbook format
ROLE_MAPPING = {
    'BUYER': 'User',
    'ASSISTANT': 'Agent',
    'QA': 'Agent',
    'QA_VENDOR': 'Agent',
    'MARKETING': 'delete',
    'SYSTEM': 'slot_update'
}

# 31 discrete Skills (no parameters) - matching design spec Appendix B
SKILL_DEFINITIONS = {
    # 通用 (General) - 8 skills
    'gen_greet': {'type': 'generation', 'description': '问候'},
    'gen_empathize': {'type': 'generation', 'description': '安抚情绪'},
    'gen_clarify': {'type': 'generation', 'description': '澄清意图'},
    'gen_verify_order': {'type': 'generation', 'description': '请求订单信息'},
    'gen_hold': {'type': 'generation', 'description': '请求等待'},
    'gen_transfer': {'type': 'generation', 'description': '转人工'},
    'gen_apologize': {'type': 'generation', 'description': '致歉'},
    'gen_close': {'type': 'generation', 'description': '结束会话'},

    # 售前 (Presale) - 7 skills
    'pre_query_product': {'type': 'preprocessing', 'description': '查询商品'},
    'pre_check_stock': {'type': 'preprocessing', 'description': '查库存'},
    'pre_compare': {'type': 'preprocessing', 'description': '商品对比'},
    'pre_recommend': {'type': 'preprocessing', 'description': '推荐'},
    'pre_answer_spec': {'type': 'preprocessing', 'description': '回答规格问题'},
    'pre_check_promo': {'type': 'preprocessing', 'description': '查优惠'},
    'pre_guide_purchase': {'type': 'preprocessing', 'description': '引导下单'},

    # 物流 (Logistics) - 7 skills
    'log_query_status': {'type': 'logging', 'description': '查物流状态'},
    'log_query_detail': {'type': 'logging', 'description': '查物流详情'},
    'log_estimate_arrival': {'type': 'logging', 'description': '预计到达时间'},
    'log_modify_address': {'type': 'logging', 'description': '修改地址'},
    'log_contact_courier': {'type': 'logging', 'description': '联系快递员'},
    'log_delay_notify': {'type': 'logging', 'description': '延迟通知'},
    'log_lost_claim': {'type': 'logging', 'description': '丢件理赔'},

    # 售后 (Aftersale) - 9 skills
    'aft_check_policy': {'type': 'action', 'description': '查退换政策'},
    'aft_collect_evidence': {'type': 'action', 'description': '收集证据'},
    'aft_initiate_refund': {'type': 'action', 'description': '发起退款'},
    'aft_initiate_return': {'type': 'action', 'description': '发起退货'},
    'aft_initiate_exchange': {'type': 'action', 'description': '发起换货'},
    'aft_schedule_pickup': {'type': 'action', 'description': '安排取件'},
    'aft_track_progress': {'type': 'action', 'description': '跟踪进度'},
    'aft_compensate': {'type': 'action', 'description': '赔偿'},
    'aft_reject_explain': {'type': 'action', 'description': '拒绝说明'},
}

# State machine transitions
STATE_MACHINE = {
    'OPEN': ['IDENTIFY'],
    'IDENTIFY': ['DIAGNOSE', 'RESOLVE', 'CLOSE'],
    'DIAGNOSE': ['RESOLVE', 'ESCALATE', 'CLOSE'],
    'RESOLVE': ['CONFIRM', 'CLOSE'],
    'CONFIRM': ['CLOSE', 'RESOLVE'],
    'ESCALATE': ['CLOSE'],
    'CLOSE': []
}

# Valid skills set
VALID_SKILLS = set(SKILL_DEFINITIONS.keys())

# Scene classification patterns
SCENE_CATEGORIES = ['presale', 'logistics', 'aftersale', 'unknown']
SCENE_PATTERNS = {
    'presale': ['购买', '下单', '价格', '优惠', '库存', '多少钱', '有没有货'],
    'logistics': ['物流', '快递', '发货', '到货', '配送', '运单', '查件'],
    'aftersale': ['退款', '退货', '换货', '投诉', '质量问题', '售后', '维修']
}