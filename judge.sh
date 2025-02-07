#!/bin/bash

export OPENAI_API_KEY="your_openai_api_key_here"

JUDGE_MODEL_PATH="openai"
TASK="GCG"
METHOD="llama2_rl_refiner"
TARGET_MODEL="llama2"
DATA_PATH="./eval/results/$TASK/$TARGET_MODEL/outputs/$METHOD.json"
SAVE_PATH="./eval/results/$TASK/$TARGET_MODEL/judged/test_JUDGED_$METHOD.json"

python ./eval/judge_responses.py --judge_model_path $JUDGE_MODEL_PATH \
                          --data_path $DATA_PATH \
                          --save_path $SAVE_PATH
