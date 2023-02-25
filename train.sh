#!/bin/bash

TASK=${1}
EXP=${2}
DEVICE=${3:-all}

echo "TASK=$TASK, EXP=$EXP, DEVICE=$DEVICE"

cd "$(dirname "${BASH_SOURCE[0]}")" &&\
git fetch --all &&\
git reset --hard "$EXP" &&\
make GPUS="'\"device=$DEVICE\"'" \
     COMMAND="python scripts/$TASK/train.py --experiment $EXP" \
     NAME="soccernet-23-$TASK-$EXP"
