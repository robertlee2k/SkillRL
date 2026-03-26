# Playbook Viewer

可视化查看 playbook 训练数据和原始 session 会话。

## 启动

```bash
cd /home/bo.li/SkillRL
uvicorn viewer.main:app --reload --port 8000
```

访问 http://localhost:8000

## 功能

1. **Playbook 列表页** (`/`)
   - 搜索 playbook_id / session_id / scenario
   - 按 scenario、subtype、has_order、sentiment 筛选
   - 点击行查看详情
   - 分页浏览

2. **Playbook 详情页** (`/playbook/{playbook_id}`)
   - 交互式流程图 (vis.js)
   - 点击节点查看详情
   - 底部显示原始 session 时间线

3. **训练监控页** (`/monitor`)
   - 统计数据概览
   - Scenario 分布
   - 技能使用频率
   - 可配置刷新间隔

## 数据源

- `outputs/playbooks_full.json` - playbook 数据
- `session_order_converted.json` - 原始 session 数据

## 依赖

```bash
pip install fastapi uvicorn pydantic
```