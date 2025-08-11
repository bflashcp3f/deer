#!/bin/bash

# bash scripts/ontonotes/run_deer_icl.sh 8 openai text-embedding-3-small togetherai Qwen/Qwen2.5-7B-Instruct-Turbo 64 1.0 1.0 0.01
# bash scripts/ontonotes/run_deer_icl.sh 8 openai text-embedding-3-small openai gpt-4o-mini-2024-07-18 64 1.0 1.0 0.01
# bash scripts/ontonotes/run_deer_icl.sh 8 openai text-embedding-3-small openai gpt-4o-2024-08-06 64 1.0 1.0 0.01

# Define variables
DATA_NAME="ontonotes"
OUTPUT_DIR="output/${DATA_NAME}"
PROMPT_TEMPLATE_NAME=icl_json_format
ICL_DEMO_NUM=$1
ICL_DEMO_RETRIEVAL_METHOD=deer
EMB_MODEL_TYPE=$2
EMB_MODEL_NAME=$3
MODEL_TYPE=$4
MODEL_NAME=$5
BATCH_SIZE=$6
ENTITY_WEIGHT=$7
CONTEXT_WEIGHT=$8
OTHER_WEIGHT=$9

python run.py \
    --icl_inference \
    --data_name $DATA_NAME \
    --eval_split test_sample_1000 \
    --prompt_template_name $PROMPT_TEMPLATE_NAME \
    --icl_demo_num $ICL_DEMO_NUM \
    --icl_demo_retrieval_method $ICL_DEMO_RETRIEVAL_METHOD \
    --emb_model_type $EMB_MODEL_TYPE \
    --emb_model_name $EMB_MODEL_NAME \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --entity_weight $ENTITY_WEIGHT \
    --context_weight $CONTEXT_WEIGHT \
    --other_weight $OTHER_WEIGHT \
    --output_dir $OUTPUT_DIR