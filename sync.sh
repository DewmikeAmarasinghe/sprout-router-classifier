#!/bin/bash
# Run at end of every Kaggle session.
set -e
git add mlruns/ results/ \
    experiments/v1/classical/ \
    src/backend/generation/examples.json \
    data/datasets/v1/train.csv \
    data/datasets/v1/val.csv \
    data/datasets/v1/test.csv \
    data/datasets/v1/quality_report.json \
    data/datasets/v1/split_stats.json
git commit -m "sync: $(date '+%Y-%m-%d %H:%M')" || echo "Nothing to commit"
git push origin main
# Transformer checkpoints → Google Drive (too large for GitHub)
cp -r experiments/v1/transformers/models/ \
    /content/drive/MyDrive/sprout-router/models/ 2>/dev/null || true
echo "Done."
