#!/usr/bin/env bash

set -x
set -e

TASK="cwq"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="=/data/${TASK}"
fi



python3 -u train.py \
--pretrained-model "microsoft/mpnet-base"\
--pooling mean \
--train_path "/data/cwq/train_goldenpath.jsonl" \
--train_graph_path "/data/cwq/total_graph_cwq.jsonl" \
--triple2id_1 "/data/cwq/cwq_triple2id.pkl" \
--batch_size 1 \
--print-freq 1000 \
--max_num_neg 50 \
--max_num_pos 20 \
--epochs 5 \
--workers 2 \
--max-to-keep 3 "$@" \
--output-dir "/ckpt/cwq"
