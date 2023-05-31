#!/bin/bash
set -e

python scripts/ball_action/train.py --experiment sampling_weights_001
python scripts/ball_action/predict.py --experiment sampling_weights_001

python scripts/action/train.py --experiment action_sampling_weights_002

python scripts/ball_action/train.py --experiment ball_tuning_001

python scripts/ball_action/train.py --experiment ball_finetune_long_004
python scripts/ball_action/predict.py --experiment ball_finetune_long_004
python scripts/ball_action/evaluate.py --experiment ball_finetune_long_004
python scripts/ball_action/predict.py --experiment ball_finetune_long_004 --challenge
python scripts/ball_action/ensemble.py --experiments ball_finetune_long_004 --challenge

cd data/ball_action/predictions/ball_finetune_long_004/challenge/ensemble/
zip results_spotting.zip ./*/*/*/results_spotting.json
