#!/bin/bash

# Set environment variables
MODEL_PATH=gpt2
DATASET=data/composition_ex_SMALL.30.3.12.6/
WEIGHT_DECAY=0.01
N_LAYERS=12
GPU=5,7
OUTPUT_DIR=/scratch/davide/model_paper/outputs_prova_checkpoint_2 # Change this to your desired output directory

# Execute the training script with the specified arguments   (they used 4 gpus I think!!)
export OMP_NUM_THREADS=2
#CUDA_VISIBLE_DEVICES=$GPU python main.py \
#CUDA_VISIBLE_DEVICES=5,7 torchrun --nproc_per_node=2 --master_port=12345 main_multy_GPU.py \
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12355 main_multy_GPU.py \
    --data_dir $DATASET \
    --model_name_or_path ${MODEL_PATH} \
    --weight_decay $WEIGHT_DECAY \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 10 \
    --max_length 10 \
    --block_size 10 \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --save_step 5000 \
    --save_step_dense 4000 \
    --max_steps 75000 \
    --do_train \
    --scheduler constant_schedule_with_warmup \
    --fp16 \
    --evaluate_during_training \
    --predict_during_training \
    --init_weights \
    --add_tokens \
    --n_layer $N_LAYERS \
    --save_best_model 
    #--overwrite_output_dir  # remember to add \ above when di-indent this option
