#!/bin/bash

python -m src.compute_avg_hidden_state --dataset_name=antonym --layer=12

python -m src.compute_avg_hidden_state --dataset_name=capitalize --layer=12

python -m src.compute_avg_hidden_state --dataset_name=country-capital --layer=12

python -m src.compute_avg_hidden_state --dataset_name=english-french --layer=12

python -m src.compute_avg_hidden_state --dataset_name=present-past --layer=12

python -m src.compute_avg_hidden_state --dataset_name=singular-plural --layer=12

# # Repeat for llama 7B
# python -m src.compute_avg_hidden_state --dataset_name=antonym --model_name=meta-llama/Llama-2-7b-hf --layer=8

# python -m src.compute_avg_hidden_state --dataset_name=capitalize --model_name=meta-llama/Llama-2-7b-hf --layer=8

# python -m src.compute_avg_hidden_state --dataset_name=country-capital --model_name=meta-llama/Llama-2-7b-hf --layer=8

# python -m src.compute_avg_hidden_state --dataset_name=english-french --model_name=meta-llama/Llama-2-7b-hf --layer=8

# python -m src.compute_avg_hidden_state --dataset_name=present-past --model_name=meta-llama/Llama-2-7b-hf --layer=8

# python -m src.compute_avg_hidden_state --dataset_name=singular-plural --model_name=meta-llama/Llama-2-7b-hf --layer=8