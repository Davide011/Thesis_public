#!/bin/bash
#SBATCH --job-name=no_2_hop_SOME_HOP      # Job name
#SBATCH --output=/scratch/davide/sbatch_logs/%x_%j.out                 # Output file with job name and ID
#SBATCH --error=/scratch/davide/sbatch_logs/%x_%j.err                  # Error file with job name and ID
#SBATCH --partition=titans                    # Specify the GPU partition
#SBATCH --gres=gpu:4                       # Request 4 GPUs
#SBATCH --cpus-per-task=10                # Number of CPU cores per task  #     replace with what is needed
#SBATCH --mem=32G                          # Memory per node
#SBATCH --time=7-00:00:00  # Default time limit (7 days)
#SBATCH --nodes=1                          # Number of nodes required
#SBATCH --mail-type=ALL                   # Send email on job start, end, and fail
#SBATCH --mail-user=s220331@student.dtu.dk   # Your email address
#SBATCH --export=ALL

#chissa se serve
source ~/.bashrc
source activate my_transformers_env

# Load necessary modules (adjust based on your environment)
#module load python/3.8m
#module load cuda/11.1                      # Load the appropriate CUDA version

# Set environment variables
MODEL_PATH=gpt2
DATASET=/home/s220331/GROK/Thesis/data_MIO_mask_NEW_correct/no_2_hop_with_some.200.20.12.6/
#/home/s220331/GROK/Thesis/data_MIO_mask_NEW_correct/no_2_hop.200.20.12.6/  
 # 
WEIGHT_DECAY=0.03             # before 0.3   now to try extream case 0.03
N_LAYERS=8                   # smaller models take more time to generalize
OUTPUT_DIR=/scratch/davide/model_paper/no_2_hop_SOME/

# Get the number of GPUs allocated by SLURM
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Execute the training script with the specified arguments
srun python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=12345 main_multy_GPU.py \
    --data_dir $DATASET \
    --model_name_or_path ${MODEL_PATH} \
    --weight_decay $WEIGHT_DECAY \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 10 \
    --max_length 10 \
    --block_size 10 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --save_step 50000 \
    --save_step_dense 40000 \
    --max_steps 1500000 \
    --do_train \
    --scheduler constant_schedule_with_warmup \
    --fp16 \
    --evaluate_during_training \
    --predict_during_training \
    --init_weights \
    --add_tokens \
    --n_layer $N_LAYERS 
    
    


echo "Done: $(date +%F-%R:%S)"