#!/usr/bin/env bash

set -x
set -e

TASK="cwq"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="/home/hyemin/model/my_model/data/${TASK}"
fi

#"microsoft/mpnet-base"

python3 -u train.py \
--pretrained-model "Alibaba-NLP/gte-large-en-v1.5" \
--pooling mean \
--train_path "/home/hyemin/model/my_model/data/cwq/train_concat_goldenpath_newtotal_0416.jsonl" \
--train_graph_path "/home/hyemin/shared_data/cwq/total_graph_cwq.jsonl" \
--triple2id_1 "/home/hyemin/shared_data/cwq/cwq_triple2id.pickle" \
--batch_size 1 \
--print-freq 1000 \
--max_num_neg 50 \
--max_num_pos 20 \
--epochs 5 \
--workers 2 \
--max-to-keep 3 "$@" \
--output-dir "/home/hyemin/model/my_model/BACKPACK/ENT_PRUNER/cwq/alibaba_totalgraph"