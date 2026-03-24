"""
LLM-based generator for scene classification and playbook generation.
Uses Doubao (compatible with OpenAI SDK) for inference.

Required environment variable:
    VOLC_API_KEY – Volcengine API key
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

# 31 valid skills (from design spec Appendix B)
VALID_SKILLS = [
    # 通用 (General) - 8 skills
    'gen_greet', 'gen_empathize', 'gen_clarify', 'gen_verify_order',
    'gen_hold', 'gen_transfer', 'gen_apologize', 'gen_close',
    # 售前 (Presale) - 7 skills
    'pre_query_product', 'pre_check_stock', 'pre_compare', 'pre_recommend',
    'pre_answer_spec', 'pre_check_promo', 'pre_guide_purchase',
    # 物流 (Logistics) - 7 skills
    'log_query_status', 'log_query_detail', 'log_estimate_arrival',
    'log_modify_address', 'log_contact_courier', 'log_delay_notify', 'log_lost_claim',
    # 售后 (Aftersale) - 9 skills
    'aft_check_policy', 'aft_collect_evidence', 'aft_initiate_refund',
    'aft_initiate_return', 'aft_initiate_exchange', 'aft_schedule_pickup',
    'aft_track_progress', 'aft_compensate', 'aft_reject_explain',
]

VALID_SKILLS_STR = ', '.join(VALID_SKILLS)


class LLMGenerator:
    """LLM-based generator using Doubao API."""

    def __init__(self):
        api_key = os.getenv('VOLC_API_KEY')
        if not api_key:
            raise ValueError(
                "[LLMGenerator] 严重错误: 找不到火山云 VOLC_API_KEY 环境变量！"
            )

        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=api_key,
        )
        # 豆包模型接入点
        self.model = "doubao-seed-2-0-pro-260215"
        logger.info(f"[LLMGenerator] Initialized with model: {self.model}")

    def call_llm_for_classification(self, conversation_text: str) -> str:
        """
        Classify conversation into presale, logistics, aftersale, or trash.

        Args:
            conversation_text: First 3 turns of conversation

        Returns:
            Scene category string: 'presale', 'logistics', 'aftersale', or 'trash'
        """
        prompt = f"""你是一个专业的电商客服对话分类专家。请根据以下对话内容判断该会话属于哪个场景。

【对话内容】:
{conversation_text}

【场景定义】:
- presale: 售前咨询（商品询问、推荐、比价、库存查询）
- logistics: 物流查询（快递、发货、配送、运单追踪）
- aftersale: 售后服务（退款、退货、换货、投诉、质量问题）
- trash: 无法分类的废数据（纯营销、无意义对话、空白对话）

请只输出一个 JSON 对象，格式为 {{"scene": "场景标签"}}，不要有任何其他文字。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=100,
            )

            content = response.choices[0].message.content
            result = json.loads(content)
            scene = result.get('scene', 'trash')

            # Validate scene is one of the allowed values
            if scene not in ['presale', 'logistics', 'aftersale', 'trash']:
                logger.warning(f"[LLMGenerator] Invalid scene returned: {scene}, defaulting to trash")
                return 'trash'

            logger.info(f"[LLMGenerator] Classification result: {scene}")
            return scene

        except json.JSONDecodeError as e:
            logger.error(f"[LLMGenerator] JSON parse error in classification: {e}")
            return 'trash'
        except Exception as e:
            logger.error(f"[LLMGenerator] Error calling LLM for classification: {e}")
            return 'trash'

    def call_llm_for_playbook(self, conversation_text: str, session_id: str = "unknown") -> Optional[Dict[str, Any]]:
        """
        Generate a complete playbook JSON from conversation text.

        Args:
            conversation_text: Full conversation with User/Agent alternation
            session_id: Session identifier for playbook_id

        Returns:
            Playbook dict following Section 5.1 schema, or None on failure
        """
        prompt = f"""你是一个专业的 RL 环境剧本构建工程师。
我将给你一段真实的电商客服二人对话记录（User 和 Agent 交替）。
你需要将其转化为强化学习环境所需的静态剧本树（JSON格式）。

【Agent 动作空间（仅限此31个，严禁捏造）】：
通用：gen_greet, gen_empathize, gen_clarify, gen_verify_order, gen_hold, gen_transfer, gen_apologize, gen_close
售前：pre_query_product, pre_check_stock, pre_compare, pre_recommend, pre_answer_spec, pre_check_promo, pre_guide_purchase
物流：log_query_status, log_query_detail, log_estimate_arrival, log_modify_address, log_contact_courier, log_delay_notify, log_lost_claim
售后：aft_check_policy, aft_collect_evidence, aft_initiate_refund, aft_initiate_return, aft_initiate_exchange, aft_schedule_pickup, aft_track_progress, aft_compensate, aft_reject_explain

【对话内容】：
{conversation_text}

🔴 【核心要求：脑补发散（Hallucinate Branches）】：
你生成的剧本必须是"树"而不是"线"！每个节点至少要有2-4个分支！

1. **提取历史真实路径**：
   - 分析Agent的每一句话，映射到上述31个Skill之一
   - Skill绝对不能带参数！只能是纯粹的skill_id

2. **🔴 强制脑补替代分支**：
   - 在每个关键节点，必须补充1-2个**合理的替代Skill**
   - 为这些替代分支生成**合理的Mock buyer_text**
   - 例如：历史是Agent先安抚情绪，脑补"如果Agent直接查物流会怎样"

3. **🔴 必须脑补负向分支**：
   - 每个剧本至少要有1-2条因Agent处理不当导致的负向路径
   - 负向路径的buyer_text必须体现买家情绪恶化（sentiment设为"angry"）
   - 例如：Agent敷衍 → 买家愤怒追问

4. **槽位更新**：
   - 如果对话中买家提供了订单号、商品ID等信息，在slot_updates中标记
   - 例如：{{"order_id_collected": true}}

5. **情绪评估**：
   - 评估User在每个节点的情绪（calm/neutral/angry）

6. **🔴 兜底节点（default_fallback）**：
   - 每个节点必须有default_fallback字段
   - 指向一个通用的"买家困惑/不满"节点
   - 当RL模型选了合法但transitions中不存在的Skill时使用

7. **合法Skill列表（available_skills）**：
   - 每个节点列出所有在该场景下合理的Skill（3-5个）

【输出格式要求】：
输出严格的JSON结构，不要有任何额外解释。
确保每个节点的transitions至少有2个分支（除了terminal节点）。

【JSON Schema】：
{{
  "nodes": {{
    "root": {{
      "buyer_text": "买家说的话",
      "sentiment": "calm|neutral|angry",
      "slot_updates": {{}},
      "available_skills": ["skill1", "skill2", ...],
      "transitions": {{
        "skill_id": "next_node_id",
        ...
      }},
      "default_fallback": "fallback_node_id"
    }},
    ...
    "terminal": {{
      "buyer_text": "[END]",
      "sentiment": "calm",
      "slot_updates": {{}},
      "available_skills": [],
      "transitions": {{}},
      "default_fallback": "terminal"
    }}
  }}
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=4096,
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            # Validate basic structure
            if 'nodes' not in result:
                logger.error("[LLMGenerator] LLM response missing 'nodes' key")
                return None

            logger.info(f"[LLMGenerator] Generated playbook with {len(result['nodes'])} nodes")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"[LLMGenerator] JSON parse error in playbook generation: {e}")
            return None
        except Exception as e:
            logger.error(f"[LLMGenerator] Error calling LLM for playbook: {e}")
            return None


# Singleton instance
_generator_instance: Optional[LLMGenerator] = None


def get_generator() -> LLMGenerator:
    """Get or create the LLM generator singleton."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = LLMGenerator()
    return _generator_instance


def call_llm_for_classification(conversation_text: str) -> str:
    """
    Classify conversation into scene category.
    Convenience function using singleton generator.
    """
    return get_generator().call_llm_for_classification(conversation_text)


def call_llm_for_playbook(conversation_text: str, session_id: str = "unknown") -> Optional[Dict[str, Any]]:
    """
    Generate playbook from conversation.
    Convenience function using singleton generator.
    """
    return get_generator().call_llm_for_playbook(conversation_text, session_id)