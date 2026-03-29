"""动作自然语言化映射器 (Action NLG Mapper)

将大模型输出的硬编码 Action 转化为友好的用户侧文案。
"""

# 31 个合法技能的自然语言映射
# 格式: action_id -> "[action_id] 中文说明"
ACTION_NLG_MAP = {
    # General skills - 通用技能
    'gen_greet': '[gen_greet] 客服向买家打招呼问好',
    'gen_empathize': '[gen_empathize] 客服对买家表示理解和共情',
    'gen_clarify': '[gen_clarify] 客服请求买家澄清需求',
    'gen_verify_order': '[gen_verify_order] 客服正在核实订单信息',
    'gen_hold': '[gen_hold] 客服请买家稍等片刻',
    'gen_transfer': '[gen_transfer] 客服正在为您转接人工服务',
    'gen_apologize': '[gen_apologize] 客服向买家致歉',
    'gen_close': '[gen_close] 客服礼貌结束对话',

    # Presale skills - 售前技能
    'pre_query_product': '[pre_query_product] 客服正在为您查询产品信息',
    'pre_check_stock': '[pre_check_stock] 客服正在为您查询库存情况',
    'pre_compare': '[pre_compare] 客服正在为您对比产品差异',
    'pre_recommend': '[pre_recommend] 客服正在为您推荐合适的产品',
    'pre_answer_spec': '[pre_answer_spec] 客服正在解答产品规格问题',
    'pre_check_promo': '[pre_check_promo] 客服正在为您查询优惠活动',
    'pre_guide_purchase': '[pre_guide_purchase] 客服正在引导您完成购买',

    # Logistics skills - 物流技能
    'log_query_status': '[log_query_status] 客服正在为您查询物流状态',
    'log_query_detail': '[log_query_detail] 客服正在为您查询物流详情',
    'log_estimate_arrival': '[log_estimate_arrival] 客服正在为您预估送达时间',
    'log_modify_address': '[log_modify_address] 客服正在帮您修改收货地址',
    'log_contact_courier': '[log_contact_courier] 客服正在联系快递员',
    'log_delay_notify': '[log_delay_notify] 客服正在通知您物流延迟信息',
    'log_lost_claim': '[log_lost_claim] 客服正在处理您的丢件理赔',

    # Aftersale skills - 售后技能
    'aft_check_policy': '[aft_check_policy] 客服正在为您查询售后政策',
    'aft_collect_evidence': '[aft_collect_evidence] 客服正在收集问题凭证',
    'aft_initiate_refund': '[aft_initiate_refund] 客服正在为您发起退款',
    'aft_initiate_return': '[aft_initiate_return] 客服正在为您发起退货',
    'aft_initiate_exchange': '[aft_initiate_exchange] 客服正在为您发起换货',
    'aft_schedule_pickup': '[aft_schedule_pickup] 客服正在为您预约上门取件',
    'aft_track_progress': '[aft_track_progress] 客服正在为您追踪售后进度',
    'aft_compensate': '[aft_compensate] 客服正在为您处理补偿',
    'aft_reject_explain': '[aft_reject_explain] 客服正在解释拒绝原因',
}


def get_action_nlg(action: str) -> str:
    """
    将动作 ID 转换为自然语言描述。

    Args:
        action: 动作 ID (如 'pre_query_product')

    Returns:
        自然语言描述，如 '[pre_query_product] 客服正在为您查询产品信息'
        如果动作未定义，返回 '[执行未定义的动作: {action}]'
    """
    return ACTION_NLG_MAP.get(action, f'[执行未定义的动作: {action}]')