#!/bin/bash
# file knowledge sharing  (== --add_recurrence), N_LAYER= 4 instead of 8

# Set environment variables
MODEL_PATH=gpt2
DATASET=/home/s220331/GROK/Thesis/data_MIO/SPLIT_composition.200.20.12.6/ # /home/s220331/GROK/Thesis/data/composition_SMALL.200.20.18.0/  #/home/s220331/GROK/Thesis/data_MIO/SPLIT_composition.200.20.12.6/
WEIGHT_DECAY=0.3
N_LAYERS=8
GPU=2,1
OUTPUT_DIR=/scratch/davide/model_paper/test_cancel_0 # Change this to your desired output directory


#CUDA_VISIBLE_DEVICES=$GPU python main.py \
#CUDA_VISIBLE_DEVICES=5,7 torchrun --nproc_per_node=2 --master_port=12345 main_multy_GPU.py \
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_multy_GPU.py \
    --data_dir $DATASET \
    --model_name_or_path ${MODEL_PATH} \
    --weight_decay $WEIGHT_DECAY \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 10 \
    --max_length 10 \
    --block_size 10 \
    --train_batch_size 50 \
    --eval_batch_size 50 \
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
    --n_layer $N_LAYERS \
    #--add_recurrence
    #--overwrite_output_dir  # remember to add \ above when di-indent this option
