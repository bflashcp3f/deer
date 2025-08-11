#!/bin/bash

# bash scripts/ncbi/run_deer_er.sh 8 deer openai text-embedding-3-small togetherai Qwen/Qwen2.5-7B-Instruct-Turbo 64 1.0 1.0 0.01 1 0.75 0.75 0.95
# bash scripts/ncbi/run_deer_er.sh 8 deer openai text-embedding-3-small openai gpt-4o-mini-2024-07-18 64 1.0 1.0 0.01 1 0.75 0.75 0.95
# bash scripts/ncbi/run_deer_er.sh 8 deer openai text-embedding-3-small openai gpt-4o-2024-08-06 64 1.0 1.0 0.01 1 0.75 0.75 0.95

# Define variables
DATA_NAME="ncbi"
OUTPUT_DIR="output/${DATA_NAME}"
ICL_DEMO_NUM=$1
ICL_DEMO_RETRIEVAL_METHOD=$2
EMB_MODEL_TYPE=$3
EMB_MODEL_NAME=$4
MODEL_TYPE=$5
MODEL_NAME=$6
BATCH_SIZE=$7
ENTITY_WEIGHT=$8
CONTEXT_WEIGHT=$9
OTHER_WEIGHT=${10}
ICL_SPAN_DEMO_NUM=${11}
ENTITY_BOUND_UNSEEN=${12}
CONTEXT_BOUND_UNSEEN=${13}
ENTITY_BOUND_FN=${14}

echo "============================================"
echo "DEER-ER: Entity Reflection Pipeline"
echo "============================================"

# Step 1: Reflect on unseen tokens
echo ""
echo "[Step 1/3] Reflecting on unseen tokens..."
echo "--------------------------------------------"
python run.py \
    --data_name $DATA_NAME \
    --eval_split test \
    --prompt_template_name reflect_unseen \
    --prior_prompt_template_name icl_json_format \
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
    --icl_span_demo_num $ICL_SPAN_DEMO_NUM \
    --entity_bound_unseen $ENTITY_BOUND_UNSEEN \
    --context_bound_unseen $CONTEXT_BOUND_UNSEEN \
    --process_abbrev unseen \
    --output_dir $OUTPUT_DIR

if [ $? -ne 0 ]; then
    echo "Error: Unseen token reflection failed"
    exit 1
fi

# Step 2: Reflecting on false negative tokens
echo ""
echo "[Step 2/3] Reflecting on false negative tokens..."
echo "--------------------------------------------"
python run.py \
    --data_name $DATA_NAME \
    --eval_split test \
    --prompt_template_name reflect_fn \
    --prior_prompt_template_name reflect_unseen \
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
    --icl_span_demo_num $ICL_SPAN_DEMO_NUM \
    --entity_bound_unseen $ENTITY_BOUND_UNSEEN \
    --context_bound_unseen $CONTEXT_BOUND_UNSEEN \
    --entity_bound_fn $ENTITY_BOUND_FN \
    --process_abbrev unseen \
    --output_dir $OUTPUT_DIR

if [ $? -ne 0 ]; then
    echo "Error: False negative token reflection failed"
    exit 1
fi

# Step 3: Reflect on boundary tokens
echo ""
echo "[Step 3/3] Reflecting on boundary tokens..."
echo "--------------------------------------------"
python run.py \
    --data_name $DATA_NAME \
    --eval_split test \
    --prompt_template_name reflect_boundary \
    --prior_prompt_template_name reflect_fn \
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
    --icl_span_demo_num $ICL_SPAN_DEMO_NUM \
    --entity_bound_unseen $ENTITY_BOUND_UNSEEN \
    --context_bound_unseen $CONTEXT_BOUND_UNSEEN \
    --entity_bound_fn $ENTITY_BOUND_FN \
    --filter_single_token_fp \
    --output_dir $OUTPUT_DIR

if [ $? -ne 0 ]; then
    echo "Error: Boundary token reflection failed"
    exit 1
fi

echo ""
echo "============================================"
echo "DEER-ER Pipeline Complete!"
echo "============================================"