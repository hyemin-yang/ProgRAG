#!/usr/bin/env bash

set -x
set -e

TASK="webqsp"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="/home/hyemin/model/my_model/data/${TASK}"
fi

#"microsoft/mpnet-base"

python3 -u train.py \
--pretrained-model "microsoft/mpnet-base" \
--pooling mean \
--train_path "/home/hyemin/model/my_model/data/webqsp/train_goldenpath_newtotal_0416.jsonl" \
--train_graph_path "/home/hyemin/shared_data/webqsp/train_max80000_subgraph.pkl" \
--triple2id_1 "/home/hyemin/shared_data/webqsp/webqsp_triple2id.pkl" \
--batch_size 1 \
--print-freq 100 \
--max_num_neg 50 \
--max_num_pos 20 \
--epochs 6 \
--workers 2 \
--max-to-keep 3 "$@" \
--output-dir "/home/hyemin/model/my_model/BACKPACK/ENT_PRUNER/webqsp/subgraph"