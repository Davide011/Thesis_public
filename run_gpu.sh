
#!/bin/bash

# Set environment variables
MODEL_PATH=gpt2
DATASET=/home/s220331/GROK/Thesis/data/composition.2000.200.18.0/  #/home/s220331/GROK/Thesis/data/composition_SMALL.200.20.18.0/        #data/composition.2000.200.12.6/  this is out side GROK!
WEIGHT_DECAY=0.01
N_LAYERS=12
GPU=2,3,4,5
#OUTPUT_DIR=/scratch/davide/model_paper/outputs_small_4gpu  # Change this to your desired output directory
OUTPUT_DIR=/scratch/davide/model_paper/outputs_data_real_test_0_4gpu

# Execute the training script with the specified arguments
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 main_multy_GPU.py \
    --data_dir $DATASET \
    --model_name_or_path ${MODEL_PATH} \
    --weight_decay $WEIGHT_DECAY \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 10 \
    --max_length 10 \
    --block_size 10 \
    --train_batch_size 25 \
    --eval_batch_size 25 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --save_step 5000 \
    --save_step_dense 4000 \
    --max_steps 50000 \
    --do_train \
    --scheduler constant_schedule_with_warmup \
    --fp16 \
    --evaluate_during_training \
    --predict_during_training \
    --init_weights \
    --add_tokens \
    --n_layer $N_LAYERS \
    #--overwrite_output_dir
