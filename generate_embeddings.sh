#!/bin/bash
set -e

QUERY_ENCODER=$1
DOCUMENT_ENCODER=$2
DATA=$3
OUTPUT_PATH=$4

# Set devices mannually
# DEVICES=(0 1 2 3 4 5 6 7)
# NUM_SHARDS=${#DEVICES[*]}

# Set devices automatically. Use all GPUs
NUM_SHARDS=`nvidia-smi -L | wc -l`
DEVICES=($(seq 0 $[NUM_SHARDS-1]))

if [ ! -d $OUTPUT_PATH/logs ]; then
    mkdir -p $OUTPUT_PATH/logs
fi

for i in $(seq 0 $[NUM_SHARDS-1]);do
    nohup python generate_embeddings.py \
        --query_encoder $QUERY_ENCODER \
        --document_encoder $DOCUMENT_ENCODER \
        --data $DATA \
        --output_path $OUTPUT_PATH/embeddings \
        --num_shards $NUM_SHARDS \
        --shard_id $i \
        --device cuda:${DEVICES[i]} \
        > $OUTPUT_PATH/logs/encoding_shard_${i}_${NUM_SHARDS}.nohup 2>&1 &
    PIDS[i]=$!
    echo "start encoding (shard $i/$NUM_SHARDS) : pid=$!" >> $OUTPUT_PATH/logs/generate_embeddings.log
    sleep 3
done

# kill all processes when the main process is killed
trap 'for i in $(seq 0 $[${#PIDS[*]}-1]);do kill ${PIDS[i]};done && echo All processes killed. && exit' SIGINT SIGTERM
wait
echo 'All shards are done.'