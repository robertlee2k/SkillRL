"""
LLM-based generator for scene classification and playbook generation.
Uses Doubao (compatible with OpenAI SDK) for inference.

Required environment variable:
    VOLC_API_KEY – Volcengine API key
"""
import os
import json
import re
import logging
import time
from typing import Dict, Any, Optional, Union
from openai import OpenAI

logger = logging.getLogger(__name__)

# LLM调用配置
LLM_TIMEOUT = 180  # 秒（增加到3分钟）
LLM_MAX_RETRIES = 2  # 重试次数
MAX_CONVERSATION_CHARS = 2500  # 对话文本最大长度（更激进截断）
MAX_TURNS_FOR_TRUNCATION = 20  # 超过此turns数时启用智能截断

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

VALID_SKILLS_SET = set(VALID_SKILLS)
VALID_SKILLS_STR = ', '.join(VALID_SKILLS)


def post_process_playbook(playbook: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process LLM-generated playbook to fix common issues.

    Fixes:
    1. Removes invalid skills from available_skills and transitions
    2. Adds fallback transitions if nodes have < 2 branches
    3. Ensures all node references exist
    4. Adds terminal node if missing
    5. Removes unreachable nodes

    Args:
        playbook: Raw playbook dict from LLM

    Returns:
        Cleaned playbook dict
    """
    if 'nodes' not in playbook:
        return playbook

    nodes = playbook['nodes']

    # Ensure terminal node exists
    if 'terminal' not in nodes:
        nodes['terminal'] = {
            'buyer_text': '[END]',
            'sentiment': 'calm',
            'slot_updates': {},
            'available_skills': [],
            'transitions': {},
            'default_fallback': 'terminal'
        }

    # Process each node
    for node_id, node in nodes.items():
        if not isinstance(node, dict):
            continue

        # Fix available_skills - filter out invalid skills
        if 'available_skills' in node:
            valid_skills = [s for s in node['available_skills'] if s in VALID_SKILLS_SET]
            if not valid_skills:
                valid_skills = ['gen_clarify', 'gen_apologize']
            node['available_skills'] = valid_skills

        # Fix transitions - filter out invalid skills and ensure ≥2 branches
        if 'transitions' in node and isinstance(node['transitions'], dict):
            # Filter invalid skills from transitions
            valid_transitions = {
                skill: target for skill, target in node['transitions'].items()
                if skill in VALID_SKILLS_SET
            }

            # Ensure targets exist
            valid_transitions = {
                skill: target for skill, target in valid_transitions.items()
                if target in nodes
            }

            # Add fallback transitions if < 2 branches (for non-terminal nodes)
            if node_id != 'terminal' and len(valid_transitions) < 2:
                # Add gen_clarify -> terminal if not present
                if 'gen_clarify' not in valid_transitions:
                    valid_transitions['gen_clarify'] = 'terminal'
                # Add gen_apologize -> terminal if still < 2
                if len(valid_transitions) < 2 and 'gen_apologize' not in valid_transitions:
                    valid_transitions['gen_apologize'] = 'terminal'

            node['transitions'] = valid_transitions

        # Ensure default_fallback points to valid node
        if 'default_fallback' not in node or node.get('default_fallback') not in nodes:
            node['default_fallback'] = 'terminal'

        # Ensure required fields exist
        if 'buyer_text' not in node:
            node['buyer_text'] = '...'
        if 'sentiment' not in node:
            node['sentiment'] = 'neutral'
        if 'slot_updates' not in node:
            node['slot_updates'] = {}

    # Remove unreachable nodes (must be reachable via transitions or default_fallback)
    reachable = {'root'}  # root is always reachable as entry point
    for node in nodes.values():
        reachable.update(node.get('transitions', {}).values())
        fallback = node.get('default_fallback', '')
        if fallback:
            reachable.add(fallback)

    # Keep only reachable nodes
    playbook['nodes'] = {k: v for k, v in nodes.items() if k in reachable}

    # Ensure at least one node has angry sentiment (required for RL punishment)
    has_angry = any(n.get('sentiment') == 'angry' for n in playbook['nodes'].values())
    if not has_angry and 'root' in playbook['nodes']:
        # Set root sentiment to angry as fallback
        playbook['nodes']['root']['sentiment'] = 'angry'

    return playbook


def extract_json_from_text(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text that may contain other content.
    Handles markdown code blocks and raw JSON.
    """
    if text is None:
        return None

    text = text.strip()
    # Try to parse directly first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code blocks
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
        r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
        r'\{[\s\S]*\}',                   # Raw JSON object
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                # Try to parse
                result = json.loads(json_str)
                return result
            except (json.JSONDecodeError, IndexError):
                # Try to fix common JSON issues
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    # Remove trailing commas
                    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                    # Fix unescaped quotes in strings (simple heuristic)
                    # Try parsing again
                    result = json.loads(json_str)
                    return result
                except:
                    continue

    # Last resort: try to find nodes object directly
    nodes_match = re.search(r'"nodes"\s*:\s*\{[\s\S]*\}', text)
    if nodes_match:
        try:
            # Wrap in a proper JSON object
            json_str = '{' + nodes_match.group(0) + '}'
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            return json.loads(json_str)
        except:
            pass

    return None


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
            timeout=LLM_TIMEOUT,
        )
        # 豆包模型接入点
        self.model = "doubao-seed-2-0-pro-260215"
        logger.info(f"[LLMGenerator] Initialized with model: {self.model}, timeout: {LLM_TIMEOUT}s")

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
                max_tokens=100,
            )

            content = response.choices[0].message.content
            if content is None:
                logger.error("[LLMGenerator] LLM returned None content")
                return 'trash'

            result = extract_json_from_text(content)

            if result is None:
                logger.error(f"[LLMGenerator] Could not parse JSON from response: {content[:100]}")
                return 'trash'

            scene = result.get('scene', 'trash')

            # Validate scene is one of the allowed values
            if scene not in ['presale', 'logistics', 'aftersale', 'trash']:
                logger.warning(f"[LLMGenerator] Invalid scene returned: {scene}, defaulting to trash")
                return 'trash'

            logger.info(f"[LLMGenerator] Classification result: {scene}")
            return scene

        except Exception as e:
            logger.error(f"[LLMGenerator] Error calling LLM for classification: {e}")
            return 'trash'

    def _truncate_conversation(self, conversation_text: str, max_chars: int = MAX_CONVERSATION_CHARS) -> str:
        """
        Truncate conversation text to prevent LLM timeout.

        For long conversations:
        - Keep first 10 turns (context establishment)
        - Keep last 10 turns (conclusion/outcome)
        - Middle section is summarized as omitted

        Args:
            conversation_text: Full conversation text
            max_chars: Maximum characters allowed

        Returns:
            Truncated conversation text
        """
        if len(conversation_text) <= max_chars:
            return conversation_text

        # Split by lines (turns)
        lines = conversation_text.split('\n')

        # If not too many turns, use simple truncation
        if len(lines) <= MAX_TURNS_FOR_TRUNCATION:
            # Simple head-tail truncation
            half_chars = max_chars // 2
            first_lines = []
            last_lines = []
            current_chars = 0

            for line in lines:
                if current_chars + len(line) + 1 > half_chars:
                    break
                first_lines.append(line)
                current_chars += len(line) + 1

            current_chars = 0
            for line in reversed(lines):
                if current_chars + len(line) + 1 > half_chars:
                    break
                last_lines.insert(0, line)
                current_chars += len(line) + 1

            truncated = '\n'.join(first_lines) + '\n...[中间部分已省略]...\n' + '\n'.join(last_lines)
            logger.info(f"[LLMGenerator] Simple truncated: {len(lines)} turns -> {len(first_lines)+len(last_lines)} turns")
            return truncated

        # Smart turn-based truncation for long conversations
        # Keep first 10 and last 10 turns
        first_turns = []
        last_turns = []
        turn_count = 0
        current_turn_lines = []

        for line in lines:
            if line.startswith('买家:') or line.startswith('客服:'):
                if current_turn_lines:
                    if turn_count < 10:
                        first_turns.extend(current_turn_lines)
                    turn_count += 1
                    current_turn_lines = []
            current_turn_lines.append(line)

        # Handle last turn
        if current_turn_lines:
            if turn_count < 10:
                first_turns.extend(current_turn_lines)
            turn_count += 1

        # Collect last 10 turns
        turn_count = 0
        current_turn_lines = []
        for line in reversed(lines):
            if line.startswith('买家:') or line.startswith('客服:'):
                if current_turn_lines:
                    if turn_count < 10:
                        last_turns[:0] = current_turn_lines
                    turn_count += 1
                    current_turn_lines = []
            current_turn_lines.insert(0, line)

        if current_turn_lines and turn_count < 10:
            last_turns[:0] = current_turn_lines

        # Build truncated text
        truncated_lines = first_turns[:50]  # Limit chars
        truncated_lines.append('...[中间对话已省略，保留关键上下文]...')
        truncated_lines.extend(last_turns[-50:] if len(last_turns) > 50 else last_turns)

        truncated = '\n'.join(truncated_lines)

        # Final char limit check
        if len(truncated) > max_chars:
            truncated = truncated[:max_chars//2] + '\n...[省略]...\n' + truncated[-max_chars//2:]

        logger.info(f"[LLMGenerator] Smart truncated: {len(lines)} lines -> {len(truncated_lines)} lines ({len(truncated)} chars)")
        return truncated

    def call_llm_for_playbook(self, conversation_text: str, session_id: str = "unknown") -> Optional[Dict[str, Any]]:
        """
        Generate a complete playbook JSON from conversation text.

        Args:
            conversation_text: Full conversation with User/Agent alternation
            session_id: Session identifier for playbook_id

        Returns:
            Playbook dict following Section 5.1 schema, or None on failure
        """
        # Truncate long conversations to prevent timeout
        truncated_text = self._truncate_conversation(conversation_text)

        prompt = f"""你是一个专业的 RL 环境剧本构建工程师。
我将给你一段真实的电商客服二人对话记录（User 和 Agent 交替）。
你需要将其转化为强化学习环境所需的静态剧本树（JSON格式）。

【Agent 动作空间（仅限此31个，严禁捏造）】：
通用：gen_greet, gen_empathize, gen_clarify, gen_verify_order, gen_hold, gen_transfer, gen_apologize, gen_close
售前：pre_query_product, pre_check_stock, pre_compare, pre_recommend, pre_answer_spec, pre_check_promo, pre_guide_purchase
物流：log_query_status, log_query_detail, log_estimate_arrival, log_modify_address, log_contact_courier, log_delay_notify, log_lost_claim
售后：aft_check_policy, aft_collect_evidence, aft_initiate_refund, aft_initiate_return, aft_initiate_exchange, aft_schedule_pickup, aft_track_progress, aft_compensate, aft_reject_explain

【对话内容】：
{truncated_text}

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

5. **🔴🔴🔴 风控兜底规则（最高优先级）**：
   - 在物流(logistics)和售后(aftersale)场景中，如果对话表明买家持续无法提供订单号，或者遇到系统查无此单的情况，你在脑补客服的应对分支时，正确的安全动作必须是使用 gen_transfer（转接人工），绝不能使用退款(aft_initiate_refund)、退换(aft_initiate_return/aft_initiate_exchange)、补偿(aft_compensate)或直接生硬拒绝(aft_reject_explain)的技能。
   - 这是防止资损的红线规则！

6. **情绪评估**：
   - 评估User在每个节点的情绪（calm/neutral/angry）

7. **🔴 兜底节点（default_fallback）**：
   - 每个节点必须有default_fallback字段
   - 指向一个通用的"买家困惑/不满"节点
   - 当RL模型选了合法但transitions中不存在的Skill时使用

8. **合法Skill列表（available_skills）**：
   - 每个节点列出所有在该场景下合理的Skill（3-5个）

【输出格式要求】：
输出严格的JSON结构，不要有任何额外解释。

🔴🔴🔴 强制要求：每个节点（除了terminal和fallback开头的节点）的transitions必须至少有2个分支！这是死规则，违反会导致验证失败！

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

        # Retry mechanism for LLM calls
        last_error = None
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                )

                content = response.choices[0].message.content

                # Debug: log response length for long conversations
                if content and len(content) > 1000:
                    logger.debug(f"[LLMGenerator] LLM returned {len(content)} chars for session {session_id}")

                result = extract_json_from_text(content)

                if result is None:
                    last_error = "Could not parse JSON from playbook response"
                    # Log the problematic response for debugging
                    if content:
                        logger.error(f"[LLMGenerator] {last_error} (attempt {attempt + 1}), response preview: {content[:500]}...")
                    else:
                        logger.error(f"[LLMGenerator] {last_error} (attempt {attempt + 1}), LLM returned empty content")
                    continue

                # Validate basic structure
                if 'nodes' not in result:
                    last_error = "LLM response missing 'nodes' key"
                    logger.error(f"[LLMGenerator] {last_error} (attempt {attempt + 1})")
                    continue

                # Post-process to fix common LLM issues
                result = post_process_playbook(result)

                logger.info(f"[LLMGenerator] Generated playbook with {len(result['nodes'])} nodes for session {session_id}")
                return result

            except Exception as e:
                last_error = str(e)
                logger.error(f"[LLMGenerator] Error calling LLM for playbook (attempt {attempt + 1}): {e}")
                if attempt < LLM_MAX_RETRIES:
                    time.sleep(2)  # Wait before retry
                    logger.info(f"[LLMGenerator] Retrying...")

        logger.error(f"[LLMGenerator] All retries failed for session {session_id}: {last_error}")
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