#!/bin/bash

#GPU=5
#CUDA_VISIBLE_DEVICES=5,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port 16365 /home/s220331/GROK/Thesis/causal_tracing_composition.py \
#python causal_tracing_composition.py \
export CUDA_LAUNCH_BLOCKING=1

#NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=4 #NUM_GPUS
#NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Execute the training script with the specified arguments
#srun python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=11345 causal_tracing_composition.py \
#CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 causal_tracing_composition.py \
python causal_tracing_composition.py \
    --dataset composition.2000.200.12.6 \
    --model_dir /dtu-compute/s220331/composition/outputs_BIG \
    --save_path /dtu-compute/s220331/composition/CAUSAL_T \
    --num_layer 8 \
    --wd 0.03