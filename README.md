# Batch-ICL

This is the code for paper 'Batch-ICL: Effective, Efficient, and Order-Agnostic In-Context Learning'

## Enviornment

```
pip install -r requirements.txt
```

Install the [transformers](https://github.com/huggingface/transformers/tree/v4.35.0) from source. Copy the contents from the 'transformers' folder we provide into the 'transformers' library folder in your environment to replace 'src/transformers/models/llama/modeling_llama.py', 'src/transformers/models/llama/`__init__`.py', and 'src/transformers/`__init__`.py'.

## Run

Run BatchICL:

```
bash Batch-ICL.sh
```

Run BatchICL without evaluating k (enumerate each layer to search for the optimal k):

```
bash best.sh
```

Run BatchICL for generation:

```
bash generation.sh
```

Run Multiple “Epochs”:

```
multi_epoch.sh
```



Modify the model size/shots and other parameters in the corresponding .sh file.

