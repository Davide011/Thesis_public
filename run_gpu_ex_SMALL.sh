#!/bin/bash

# Set environment variables
MODEL_PATH=gpt2
DATASET=data/composition_ex_SMALL.30.3.12.6/         #data/composition.2000.200.12.6/  this is out side GROK!
WEIGHT_DECAY=0.01
N_LAYERS=12
GPU=3,4,5
OUTPUT_DIR=~/GROK/Thesis/outputs_ex_0  # Change this to your desired output directory

# Execute the training script with the specified arguments
CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --data_dir $DATASET \
    --model_name_or_path ${MODEL_PATH} \
    --weight_decay $WEIGHT_DECAY \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 10 \
    --max_length 10 \
    --block_size 10 \
    --train_batch_size 100 \
    --eval_batch_size 100 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --save_step 500 \
    --save_step_dense 400 \
    --max_steps 15000 \
    --do_train \
    --scheduler constant_schedule_with_warmup \
    --fp16 \
    --evaluate_during_training \
    --predict_during_training \
    --init_weights \
    --add_tokens \
    --n_layer $N_LAYERS 
    #--overwrite_output_dir  # remember to add \ above when di-indent this option
