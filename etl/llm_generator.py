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


# 安全话术：不推进业务但不会激怒买家（允许作为幽灵动作存在）
SAFE_FALLBACK_SKILLS = {
    'gen_clarify',    # 澄清意图
    'gen_empathize',  # 安抚共情
    'gen_greet',      # 寒暄打招呼
    'gen_apologize',  # 道歉
    'gen_hold'        # 请稍等
}

# 业务强制绑定映射：当 slots 中存在特定槽位时，必须添加的技能和对应的目标节点
# 关键：这些技能必须同时在 available_skills 和 transitions 中添加！
SLOT_BINDINGS = {
    'invoice_requested': {
        'skill': 'aft_issue_invoice',
        'default_target': 'terminal'  # 发票开具后通常结束对话
    },
    # 未来可扩展其他强制绑定技能
}


def post_process_playbook(playbook: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process LLM-generated playbook to fix common issues.

    Fixes:
    1. Removes invalid skills from available_skills and transitions
    2. Adds fallback transitions if nodes have < 2 branches
    3. Ensures all node references exist
    4. Adds terminal node if missing
    5. Removes unreachable nodes
    6. [CRITICAL] Ensures available_skills = transitions.keys() | SAFE_FALLBACK_SKILLS
    7. [CRITICAL] Adds forced skills based on slot bindings (e.g., invoice)

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

    # Get initial slots for dynamic patching
    initial_slots = playbook.get('initial_slots', {})

    # Process each node
    for node_id, node in nodes.items():
        if not isinstance(node, dict):
            continue

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

    # ============================================================
    # [CRITICAL FIX] Remove ghost actions & Add forced skills
    # [CRITICAL] Use DFS traversal to propagate accumulated slots!
    # ============================================================
    visited = set()  # Track processed nodes (also used for unreachable detection)

    def dfs_process_node(node_id: str, accumulated_slots: Dict[str, Any]) -> None:
        """
        DFS traversal: process node with accumulated slots, then traverse children.

        Args:
            node_id: Current node ID
            accumulated_slots: Slots accumulated from root path (NOT including this node yet)
        """
        if node_id in visited or node_id not in nodes:
            return

        visited.add(node_id)
        node = nodes[node_id]

        # Merge current node's slot_updates into accumulated_slots
        current_slots = accumulated_slots.copy()
        current_slots.update(node.get('slot_updates', {}))

        transitions = node.get('transitions', {})
        transition_keys = set(transitions.keys())

        # Dynamic patching: Add forced skills based on accumulated slots
        for slot_name, binding in SLOT_BINDINGS.items():
            if current_slots.get(slot_name):
                skill = binding['skill']
                default_target = binding['default_target']

                # CRITICAL: Must add to BOTH transitions AND available_skills!
                if skill not in transition_keys:
                    transitions[skill] = default_target
                    transition_keys.add(skill)

        # Update transitions (may have been modified by dynamic patching)
        node['transitions'] = transitions

        # [CRITICAL] available_skills = transitions.keys() | SAFE_FALLBACK_SKILLS
        # This ensures NO ghost business actions exist!
        correct_available = transition_keys | SAFE_FALLBACK_SKILLS
        node['available_skills'] = list(correct_available)

        # DFS: traverse children via transitions edges
        # Pass current_slots (including this node's updates) to children
        for target_node_id in transitions.values():
            if target_node_id and target_node_id != node_id:  # Avoid self-loops
                dfs_process_node(target_node_id, current_slots)

    # Start DFS from root with initial_slots
    dfs_process_node('root', initial_slots.copy())

    # Handle orphan nodes (not reachable from root) - process with initial_slots only
    for node_id in nodes:
        if node_id not in visited:
            node = nodes[node_id]
            transitions = node.get('transitions', {})
            transition_keys = set(transitions.keys())

            # Orphan nodes: only use initial_slots (no accumulated context)
            for slot_name, binding in SLOT_BINDINGS.items():
                if initial_slots.get(slot_name):
                    skill = binding['skill']
                    default_target = binding['default_target']
                    if skill not in transition_keys:
                        transitions[skill] = default_target
                        transition_keys.add(skill)

            node['transitions'] = transitions
            correct_available = transition_keys | SAFE_FALLBACK_SKILLS
            node['available_skills'] = list(correct_available)

    # Remove unreachable nodes: use visited set from DFS
    playbook['nodes'] = {k: v for k, v in nodes.items() if k in visited}

    # [CRITICAL FIX] Re-fix default_fallback after node removal
    # Some nodes may have default_fallback pointing to nodes that were just removed
    for node_id, node in playbook['nodes'].items():
        fallback = node.get('default_fallback', '')
        if fallback not in playbook['nodes']:
            node['default_fallback'] = 'terminal'

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

4. **动态业务上下文提取（开放式信息抽取）**：
   - **抛弃固有槽位**，采用"开放域抽象信息抽取"方法，提取对话中产生的所有隐性业务知识
   - 必须提取以下5个抽象维度（均为自然语言文本，不是布尔标记）：

   a) **user_intent（买家核心诉求）**：
      - 买家此次咨询的真实目的（如："想确认赠品是否为bebebus滑板车"、"投诉发票金额错误"、"催促发货"）

   b) **item_specifics（商品具体信息）**：
      - 买家明确提及或确认的商品细节（如："确认购买的是藏青色M码"、"询问的型号是A7Pro"、"预算范围300-500元"）

   c) **system_status（系统状态确认）**：
      - 客服查询后告知的物流/订单状态（如："物流卡在义乌分拨中心3天"、"订单号显示已签收但买家未收到"、"退差价申请已提交"）

   d) **agent_commitments（客服承诺与约束）**：
      - 客服明确给出的承诺或特殊处理方案（如："承诺24小时内发货"、"约定超期可退货"、"确认会追加赠品"）

   e) **other_crucial_context（兜底关键信息）**：
      - 【防丢弃机制】：如果对话中出现了极其重要、决定后续走向，但实在无法合理归入前4类的业务上下文（例如：前置条件"等待买家晚点发破损照片"、买家强烈的预算底线等），必须存入此处。如果没有，可留空。

   🔴 **【边界防混淆规则（极度重要）】**：
   - `system_status` 仅限不可更改的客观系统事实（如：系统显示缺货、物流停滞）。
   - `agent_commitments` 包含主观的人为干预、特殊放宽或动作承诺（如：答应退差价、答应从其他仓调货）。
   - **冲突仲裁**：如果一句话同时包含系统状态和客服承诺（例如："查了系统缺货，但我答应您明天从别处调货补发"），**优先且必须整体归入 `agent_commitments`**！

   🔴 **【状态流转与增量更新规则（DST Rules）】**：
   1. **增量输出 (Delta Updates)**：`slot_updates` 代表的是"当前这一轮节点"产生的状态**变化量**。由于子节点会自动继承父节点的状态，如果某个维度在当前轮次没有新信息或变化，**请不要输出该字段（留空即可）**，避免冗余。
   2. **覆写与清除 (Overwrite & Clear)**：如果买家在当前轮次改变了主意（如：原来要红色的，现在说要蓝色的），请直接输出 `"item_specifics": "买家改要蓝色款"`，下游系统会自动覆写旧值。如果某个诉求已经被彻底解决且后续不再需要关注，可以输出 `"user_intent": "已解决"` 来刷新状态。
   3. **多值合并 (Multi-value Concat)**：如果买家在同一句话中表达了多个意图（如既问发票又催物流），或者锁定了多个商品，请不要遗漏，将它们用分号合并在一个字符串中。例如：`"user_intent": "催促发货; 询问发票开具流程"`。

   - 这些动态上下文将作为状态机的补丁，随着对话树向下流转，成为后续 Agent 继续服务时必须知道的客观环境。

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
      "slot_updates": {{
        "user_intent": "买家核心诉求（自然语言描述）",
        "item_specifics": "商品具体信息（型号/颜色/尺寸等）",
        "system_status": "物流/订单状态确认",
        "agent_commitments": "客服承诺内容",
        "other_crucial_context": "等待买家后续补充破损图片（可选的兜底信息）"
      }},
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