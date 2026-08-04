#!/bin/bash

# Set environment variables
MODEL_PATH=/scratch/davide/model_paper/outputs_OOD_MODIFIED_composition_SMALL.200.20.18.0/checkpoint-350000/  #gpt2
DATASET=/home/s220331/GROK/Thesis/data/OOD_MODIFIED_composition_SMALL.200.20.18.0/ #/home/s220331/GROK/Thesis/data/composition.2000.200.18.0/  #/home/s220331/GROK/Thesis/data/composition_SMALL.200.20.18.0/        #data/composition.2000.200.12.6/  this is out side GROK!
WEIGHT_DECAY=0.03      # higher weight decay accellerate grokking

python inference_script.py \
    --data_dir $DATASET \
    --model_name_or_path $MODEL_PATH \ 
    --do_predict \
    --prediction_dir ./predictions \
    --custom_test test.json