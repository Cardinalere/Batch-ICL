export MODEL_PATH_7='meta-llama/Llama-2-7b-hf'
export MODEL_PATH_13='meta-llama/Llama-2-13b-hf'


python Batch-ICL_best.py \
    --ckpt_dir $MODEL_PATH_13 \
    --param_size 13 \
    --model_type llama \
    --ntrain 4


