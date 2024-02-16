export MODEL_PATH_7='meta-llama/Llama-2-7b-hf'
export MODEL_PATH_13='meta-llama/Llama-2-13b-hf'

python BatchICL_generation.py \
    --param_size 7 \
    --model_type llama \
    --ckpt_dir $MODEL_PATH_7 \
    --ntrain 4