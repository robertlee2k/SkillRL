#!/usr/bin/env python
"""
业务知识模式挖掘脚本 (Business Context Pattern Discovery)

通过开放式信息抽取 (Open IE)，从真实客服对话中挖掘潜在的业务知识模式。

Usage:
    python scripts/discover_slot_patterns.py \
        --input /path/to/sessions.json \
        --sample-size 100 \
        --output outputs/slot_patterns_report.md
"""

import os
import sys
import json
import argparse
import logging
import random
import re
from typing import Dict, Any, List, Optional
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# LLM 配置
LLM_TIMEOUT = 120
LLM_MAX_RETRIES = 2


class SlotPatternDiscovery:
    """开放式业务知识模式挖掘器"""

    def __init__(self):
        api_key = os.getenv('VOLC_API_KEY')
        if not api_key:
            raise ValueError("找不到火山云 VOLC_API_KEY 环境变量！")

        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=api_key,
            timeout=LLM_TIMEOUT,
        )
        self.model = "doubao-seed-2-0-pro-260215"
        logger.info(f"Initialized with model: {self.model}")

    def load_sessions(self, file_path: str) -> List[Dict[str, Any]]:
        """加载原始会话数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 处理不同的数据格式
        if isinstance(data, dict) and 'data' in data:
            sessions = data['data']
        elif isinstance(data, list):
            sessions = data
        else:
            sessions = [data]

        logger.info(f"Loaded {len(sessions)} sessions from {file_path}")
        return sessions

    def filter_complex_sessions(
        self,
        sessions: List[Dict[str, Any]],
        min_turns: int = 4,
        min_messages: int = 8
    ) -> List[Dict[str, Any]]:
        """筛选复杂对话（多轮交互）"""
        complex_sessions = []

        for session in sessions:
            # 检查 messages 字段
            messages = session.get('messages', [])
            if len(messages) >= min_messages:
                complex_sessions.append(session)
                continue

            # 检查 turns 字段
            turns = session.get('turns', [])
            if len(turns) >= min_turns:
                complex_sessions.append(session)
                continue

        logger.info(f"Filtered {len(complex_sessions)} complex sessions (min {min_messages} messages or {min_turns} turns)")
        return complex_sessions

    def sample_sessions(
        self,
        sessions: List[Dict[str, Any]],
        sample_size: int,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """均匀随机抽样"""
        random.seed(seed)

        if len(sessions) <= sample_size:
            return sessions

        return random.sample(sessions, sample_size)

    def format_conversation(self, session: Dict[str, Any]) -> str:
        """格式化会话为对话文本"""
        lines = []

        # 处理 messages 格式
        messages = session.get('messages', [])
        if messages:
            for msg in messages:
                sender = msg.get('sent_by', 'UNKNOWN')
                content = msg.get('content', '')
                fmt = msg.get('format', 'TEXT')

                # 跳过图片链接
                if fmt == 'IMAGE' and 'http' in content:
                    content = '[图片]'

                role_map = {
                    'BUYER': '买家',
                    'ASSISTANT': '客服',
                    'QA': '系统QA',
                    'SYSTEM': '系统'
                }
                role = role_map.get(sender, sender)
                lines.append(f"{role}: {content}")

        # 处理 turns 格式
        turns = session.get('turns', [])
        if turns and not messages:
            for turn in turns:
                role = turn.get('role', 'Unknown')
                text = turn.get('text', '')
                role_map = {'User': '买家', 'Agent': '客服'}
                role = role_map.get(role, role)
                lines.append(f"{role}: {text}")

        return '\n'.join(lines)

    def extract_entities(self, conversation: str, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """调用 LLM 进行开放式信息抽取"""

        prompt = f"""你是一个专业的数据挖掘专家。请阅读以下电商客服的真实对话记录。
你的任务是：提取出这段对话中产生的所有"隐性业务知识"和"关键上下文"。
这些知识是后续客服（或AI Agent）继续服务买家时，必须知道的客观信息。

请抛弃传统的固有槽位，尽最大可能发散提取！包括但不限于：
- 订单与物流状态（如：单号、快递公司、卡在某个分拨中心、预计到达时间）
- 商品具体信息（如：买家发图确认的是某款特定型号、颜色、尺寸、价格）
- 优惠与政策确认（如：客服承诺了退差价、承诺了送特定赠品、确认了超期退货规则）
- 核心矛盾点（如：买家核心诉求是发票开错了、赠品漏发了、收货地址错误）
- 买家画像与偏好（如：买家的预算范围、品牌偏好、紧急程度）
- 客服承诺与约束（如：客服承诺的响应时效、特殊处理方案）

【对话内容】：
{conversation}

请以 JSON 数组的形式输出你提取的所有业务实体，格式如下：
[
  {{"entity_key": "物流异常状态", "entity_value": "卡在义乌分拨中心", "description": "客服查询后告知买家的物流停滞原因"}},
  {{"entity_key": "确认赠品内容", "entity_value": "bebebus滑板车", "description": "客服解答活动时承诺的具体赠品"}}
]

如果没有提取到明显的业务实体，返回空数组 []。
只输出 JSON 数组，不要有任何其他文字。"""

        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.3,
                )

                content = response.choices[0].message.content
                if not content:
                    return None

                # 解析 JSON
                # 尝试提取 JSON 数组
                json_match = re.search(r'\[[\s\S]*\]', content)
                if json_match:
                    entities = json.loads(json_match.group())
                    return entities
                else:
                    # 尝试直接解析
                    return json.loads(content)

            except json.JSONDecodeError as e:
                logger.warning(f"[{session_id}] JSON parse error (attempt {attempt+1}): {e}")
                if attempt < LLM_MAX_RETRIES:
                    continue
            except Exception as e:
                logger.error(f"[{session_id}] LLM call error (attempt {attempt+1}): {e}")
                if attempt < LLM_MAX_RETRIES:
                    continue

        return None

    def cluster_entities(
        self,
        all_entities: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """聚类相似的 entity_key"""

        # 统计 entity_key 频次
        key_counter = Counter(e['entity_key'] for e in all_entities if 'entity_key' in e)

        # 使用 LLM 进行聚类
        unique_keys = list(key_counter.keys())
        if len(unique_keys) < 5:
            return {k: [] for k in unique_keys}

        cluster_prompt = f"""你是一个数据分类专家。以下是从电商客服对话中提取出的业务实体键名列表。
请将这些键名聚类成 5-8 个有意义的业务类别。

【实体键名列表】：
{json.dumps(unique_keys, ensure_ascii=False, indent=2)}

请输出一个 JSON 对象，格式为：
{{
  "物流状态类": ["物流异常状态", "快递单号", ...],
  "商品信息类": ["商品型号", "商品颜色", ...],
  ...
}}

只输出 JSON 对象，不要有任何其他文字。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": cluster_prompt}],
                max_tokens=2000,
                temperature=0.1,
            )

            content = response.choices[0].message.content
            if content:
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"Clustering error: {e}")

        # 如果聚类失败，使用简单的关键词分组
        return self._fallback_cluster(unique_keys)

    def _fallback_cluster(self, keys: List[str]) -> Dict[str, List[str]]:
        """简单的关键词分组作为 fallback"""
        clusters = defaultdict(list)

        keywords_map = {
            '物流状态类': ['物流', '快递', '发货', '收货', '配送', '运单', '分拨'],
            '商品信息类': ['商品', '型号', '颜色', '尺寸', '规格', '库存', '价格'],
            '优惠活动类': ['优惠', '活动', '赠品', '促销', '折扣', '满减', '券'],
            '订单相关类': ['订单', '单号', '下单', '支付', '退款', '退货'],
            '售后问题类': ['售后', '质量', '问题', '投诉', '维修', '换货'],
            '买家信息类': ['买家', '地址', '电话', '收件', '发票'],
            '客服承诺类': ['承诺', '约定', '回复', '处理', '时效'],
        }

        for key in keys:
            clustered = False
            for cluster_name, keywords in keywords_map.items():
                if any(kw in key for kw in keywords):
                    clusters[cluster_name].append(key)
                    clustered = True
                    break
            if not clustered:
                clusters['其他'].append(key)

        return dict(clusters)

    def generate_report(
        self,
        results: List[Dict[str, Any]],
        clusters: Dict[str, List[str]],
        output_path: str
    ) -> None:
        """生成 Markdown 报告"""

        # 统计
        entity_counter = Counter()
        for r in results:
            for e in r.get('entities', []):
                if 'entity_key' in e:
                    entity_counter[e['entity_key']] += 1

        # 生成报告
        report_lines = [
            "# 业务知识模式挖掘报告",
            f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## 统计概览\n",
            f"- 分析会话数: {len(results)}",
            f"- 提取实体总数: {sum(entity_counter.values())}",
            f"- 唯一实体类型: {len(entity_counter)}",
            f"\n## 实体频次排行 (Top 20)\n",
            "| 排名 | 实体键名 | 出现次数 |",
            "|------|----------|----------|",
        ]

        for i, (key, count) in enumerate(entity_counter.most_common(20), 1):
            report_lines.append(f"| {i} | {key} | {count} |")

        # 聚类结果
        report_lines.extend([
            "\n## 业务聚类结果\n",
        ])

        for cluster_name, keys in clusters.items():
            report_lines.append(f"\n### {cluster_name}\n")

            # 找出该聚类下的实体示例
            examples = []
            for r in results:
                for e in r.get('entities', []):
                    if e.get('entity_key') in keys:
                        examples.append({
                            'session_id': r.get('session_id', 'unknown'),
                            'entity_key': e.get('entity_key'),
                            'entity_value': e.get('entity_value', ''),
                            'description': e.get('description', '')
                        })

            # 只显示前 5 个示例
            for ex in examples[:5]:
                report_lines.append(f"- **{ex['entity_key']}**: `{ex['entity_value']}`")
                if ex['description']:
                    report_lines.append(f"  - {ex['description']}")
                report_lines.append(f"  - 来源: session `{ex['session_id']}`\n")

            if len(examples) > 5:
                report_lines.append(f"  - ...还有 {len(examples) - 5} 个示例\n")

        # 详细结果
        report_lines.extend([
            "\n## 详细提取结果\n",
            "<details>",
            "<summary>点击展开全部提取结果</summary>\n",
        ])

        for r in results:
            session_id = r.get('session_id', 'unknown')
            entities = r.get('entities', [])
            if entities:
                report_lines.append(f"\n### Session: {session_id}\n")
                for e in entities:
                    report_lines.append(f"- **{e.get('entity_key', 'N/A')}**: `{e.get('entity_value', 'N/A')}`")

        report_lines.extend([
            "\n</details>",
        ])

        # 写入文件
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='业务知识模式挖掘')
    parser.add_argument('--input', type=str,
                        default='/home/bo.li/data/SkillRL/session_order_converted.json',
                        help='原始会话数据文件路径')
    parser.add_argument('--sample-size', type=int, default=100,
                        help='抽样数量')
    parser.add_argument('--output', type=str,
                        default='outputs/slot_patterns_report.md',
                        help='输出报告路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("业务知识模式挖掘脚本启动")
    logger.info("=" * 60)

    # 初始化
    discovery = SlotPatternDiscovery()

    # 加载数据
    sessions = discovery.load_sessions(args.input)

    # 筛选复杂对话
    complex_sessions = discovery.filter_complex_sessions(sessions)

    # 抽样
    sampled = discovery.sample_sessions(complex_sessions, args.sample_size, args.seed)

    # 提取实体
    all_entities = []
    results = []

    for i, session in enumerate(sampled):
        session_id = session.get('session_id', f'unknown_{i}')

        # 格式化对话
        conversation = discovery.format_conversation(session)

        if len(conversation) < 50:
            logger.warning(f"[{session_id}] Conversation too short, skipping")
            continue

        logger.info(f"[{i+1}/{len(sampled)}] Processing session {session_id}...")

        # 提取实体
        entities = discovery.extract_entities(conversation, session_id)

        if entities:
            all_entities.extend(entities)
            results.append({
                'session_id': session_id,
                'entities': entities,
                'conversation_length': len(conversation)
            })
            logger.info(f"  Extracted {len(entities)} entities")
        else:
            results.append({
                'session_id': session_id,
                'entities': [],
                'conversation_length': len(conversation)
            })

    logger.info(f"\nTotal entities extracted: {len(all_entities)}")

    # 聚类
    logger.info("\nClustering entities...")
    clusters = discovery.cluster_entities(all_entities)

    # 生成报告
    discovery.generate_report(results, clusters, args.output)

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()