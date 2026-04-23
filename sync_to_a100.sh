#!/bin/bash
# Sync local SkillRL to A100 server
# Usage: ./sync_to_a100.sh [--dry-run]

REMOTE="a100:/home/bo.li/SkillRL/"
LOCAL="/Users/sherry/20-python/SkillRL/"

ARGS="-avz --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude '.idea/' \
  --exclude '.superpowers/' \
  --exclude 'wandb/' \
  --exclude '*.log'"

if [ "$1" = "--dry-run" ]; then
    echo "=== Dry run (no changes will be made) ==="
    rsync $ARGS --dry-run "$LOCAL" "$REMOTE"
else
    echo "=== Syncing to A100 ==="
    rsync $ARGS "$LOCAL" "$REMOTE"
    echo "=== Done ==="
fi
