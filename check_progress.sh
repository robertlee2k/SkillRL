#!/bin/bash
# 检查 ETL Pipeline 进度

echo "=== ETL Pipeline 进度 ==="
echo ""

# 检查输出文件
if [ -f "outputs/playbooks_all_v2.json" ]; then
    count=$(python -c "import json; print(len(json.load(open('outputs/playbooks_all_v2.json'))))")
    echo "✓ 已生成 playbook: $count 个"
    echo ""

    # 场景分布
    echo "场景分布:"
    python -c "
import json
with open('outputs/playbooks_all_v2.json') as f:
    pbs = json.load(f)
scenarios = {}
for pb in pbs:
    s = pb.get('scenario', 'unknown')
    scenarios[s] = scenarios.get(s, 0) + 1
for s, c in sorted(scenarios.items(), key=lambda x: -x[1]):
    print(f'  {s}: {c}')
"
else
    echo "输出文件还未创建"
fi

echo ""

# 检查进程
pid=$(pgrep -f "run_etl.py" | head -1)
if [ -n "$pid" ]; then
    workers=$(pgrep -f "run_etl.py" | wc -l)
    echo "✓ Pipeline 进程运行中 (主进程 PID: $pid, $workers 个进程)"
else
    echo "✗ Pipeline 进程未运行"
fi

# 从日志读取最新进度
for logfile in "/tmp/parallel_etl_20.log" "/tmp/parallel_etl.log"; do
    if [ -f "$logfile" ]; then
        latest=$(grep "Progress:" "$logfile" | tail -1)
        if [ -n "$latest" ]; then
            echo ""
            echo "最新进度:"
            echo "  $latest"
            break
        fi
    fi
done

echo ""
echo "========================"
echo ""
echo "功能说明:"
echo "  - 每 50 条 playbook 自动保存一次"
echo "  - 支持断点续传（重新运行会跳过已处理的）"
echo "  - 中途崩溃不会丢失进度"
echo "  - 并行处理 (10 workers)"