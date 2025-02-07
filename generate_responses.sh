#!/bin/bash

export OPENAI_API_KEY="your_openai_api_key_here"

TARGET_MODEL_PATH="/data/Llama-2-7b-chat-hf"
TARGET_MODEL="llama2"
METHOD="Refiner"
RFM_CONFIG_PATH="/data/rl_gemma_llama2/90"
RFM_WEIGHTS_PATH="/data/rl_gemma_llama2/90"
DATA_PATH="./data/attack_samples/llama2_gcg.json"
TASK="GCG"
SAVE_PATH="./eval/results/$TASK/$TARGET_MODEL/outputs/llama2_rl_refiner.json"

python ./eval/generate_response.py --target_model_path $TARGET_MODEL_PATH \
									 --method $METHOD \
                                     --rfm_config_path $RFM_CONFIG_PATH \
                                     --rfm_weights_path $RFM_WEIGHTS_PATH \
                                     --data_path $DATA_PATH \
                                     --task $TASK \
                                     --save_path $SAVE_PATH