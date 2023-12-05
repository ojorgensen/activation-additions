#!/bin/bash

python -m src.compute_avg_hidden_state --dataset_name=antonym --layers 13 --model_name=EleutherAI/gpt-neox-20b --training_set_length 100




# # Repeat for llama 7B
# python -m src.compute_avg_hidden_state --dataset_name=antonym --model_name=meta-llama/Llama-2-7b-hf --layers 7 8 9 10 11 12

# python -m src.compute_avg_hidden_state --dataset_name=capitalize --model_name=meta-llama/Llama-2-7b-hf --layers 7 8 9 10 11 12

# python -m src.compute_avg_hidden_state --dataset_name=country-capital --model_name=meta-llama/Llama-2-7b-hf --layers 7 8 9 10 11 12

# python -m src.compute_avg_hidden_state --dataset_name=english-french --model_name=meta-llama/Llama-2-7b-hf --layers 7 8 9 10 11 12

# python -m src.compute_avg_hidden_state --dataset_name=present-past --model_name=meta-llama/Llama-2-7b-hf --layers 7 8 9 10 11 12

# python -m src.compute_avg_hidden_state --dataset_name=singular-plural --model_name=meta-llama/Llama-2-7b-hf --layers 7 8 9 10 11 12