#!/bin/bash

export OPENAI_API_KEY="your_openai_api_key_here"

TARGET_MODEL="chatgpt"
MODEL_A_NAME="rl_refiner"
MODEL_B_NAME="no_defense"
OUTPUTS_PATH_A=""
OUTPUTS_PATH_B=""
SAVE_PATH=""

python ./eval/pairwise_winrate.py \
  --target_model $TARGET_MODEL \
  --model_a_name $MODEL_A_NAME \
  --model_b_name $MODEL_B_NAME \
  --outputs_path_a $OUTPUTS_PATH_A \
  --outputs_path_b $OUTPUTS_PATH_B \
  --save_path $SAVE_PATH
