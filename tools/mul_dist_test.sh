#!/usr/bin/env bash
# test one model 
PYTHON=${PYTHON:-"python"}

CONFIG=$1
# CHECKPOINT=$2
GPUS=$3
INIT_EPOCH=$4
MAX_EPOCH=$5
INTERVAL=$6
PORT=${PORT:-29500}

for (( i=$INIT_EPOCH; i<=$MAX_EPOCH; i=i+$INTERVAL))
do
    CHECKPOINT=$2"epoch_${i}.pth"
    echo "########################################   epoch_${i}.pth   ########################################"
    $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:7}
done

mv /data/xiongweiyu/mmdet/data_df.csv /data/xiongweiyu/mmdet/$(basename $2).csv
echo "batch dist test finish"