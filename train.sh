#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH="/data/sft_gemma-2b/final_checkpoint"
ACTOR_TOKENIZER_PATH="/data/sft_gemma-2b/checkpoint-600"
CRITIC_MODEL_PATH="/data/model/sft_gemma-2b/final_checkpoint"
PROTECTED_MODEL_PATH="/data/model/vicuna-7b-v1.5"
QUALITY_REWARD_MODEL_PATH="/data/model/RM-Gemma-2B"
SAFETY_REWARD_MODEL_PATH="/data/model/Meta-Llama-Guard-2-8B"
OUTPUT="./rl_refinement"
DATA_PATH="./data/rl_refine.json"
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=3
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
mkdir -p $OUTPUT

Num_Padding_at_Beginning=0 # this is model related

Actor_Lr=7e-7
Critic_Lr=9e-6

deepspeed --master_port 12346 ./custom_dschat/main.py \
   --data_path $DATA_PATH \
   --data_split 0,0,10 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --actor_model_tokenizer_path $ACTOR_TOKENIZER_PATH \
   --protected_model_name_or_path $PROTECTED_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --quality_reward_model_name_or_path $QUALITY_REWARD_MODEL_PATH \
   --safety_reward_model_name_or_path $SAFETY_REWARD_MODEL_PATH \
   --num_padding_at_beginning 0 \
   --per_device_generation_batch_size 2 \
   --per_device_training_batch_size 4 \
   --generation_batches 1 \
   --ppo_epochs 3 \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT/tensorboard \
   --max_answer_seq_len 1024 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type linear \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --num_warmup_steps 5 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --protected_zero_stage 2 \
   --print_answers \
   --dtype "bf16" \
   --saving_steps 30 \
   --output_dir $OUTPUT \
    &> $OUTPUT/training.log
