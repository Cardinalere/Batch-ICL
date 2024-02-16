export MODEL_PATH_7='meta-llama/Llama-2-7b-hf'
export MODEL_PATH_13='meta-llama/Llama-2-13b-hf'


python Batch-ICL-multi_epoch.py \
    --ckpt_dir $MODEL_PATH_7 \
    --param_size 7 \
    --model_type llama \
    --ntrain 4 \
    --epoch 2

