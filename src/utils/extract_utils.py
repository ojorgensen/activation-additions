import os, re, json

from typing import List, Dict

import torch
import numpy as np
import pandas as pd
from baukit import TraceDict


# Activations
def gather_activations(prompt, layers, model, tokenizer):
    """
    Collects activations for arbitrary prompts

    Parameters:
    prompt string to be passed to model
    layers: layer names to get activatons from
    model: huggingface model
    tokenizer: huggingface tokenizer

    Returns:
    td: tracedict with stored activations
    """
    sentence = [prompt]

    inputs = tokenizer(sentence, return_tensors='pt').to(model.device)

    # Access Activations 
    with TraceDict(model, layers=layers, retain_input=True, retain_output=False) as td:                
        model(**inputs) # batch_size x n_tokens x vocab_size, only want last token prediction

    return td

def gather_activations_from_dataset(
        dataset: List[str], 
        activation_types: List[str], 
        model, 
        tokenizer, 
        model_config: Dict, 
        N_TRIALS: int, 
        split_attention: bool=False,
        final_activations_only: bool=False,
        ):
    """
    Collects all desired activations for arbitrary prompts, for a given type of activation.
    Returns a dictionary of tensors, with keys corresponding to the activation types.
    Arguments:
    dataset: list of strings to be passed to model
    activation_types: list of activation types to be collected
    model: huggingface model
    tokenizer: huggingface tokenizer
    model_config: dictionary of model configuration
    N_TRIALS: number of samples to collect
    split_attention: whether to split attention tensors into heads
    """
    def split_activations_by_head(activations, model_config):
        """
        Splits the output of the attention layer into heads
        TODO: understand how this works!
        """
        new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
        activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
        return activations


    is_llama = 'llama' in model_config['name_or_path']
    prepend_bos = not is_llama
    activation_storage = {}

    # Get all layer activations we want
    all_activation_layers = []
    for activation_type in activation_types:
        all_activation_layers += model_config[activation_type]
        activation_storage[activation_type] = torch.zeros(N_TRIALS, len(model_config[activation_type]), model_config['resid_dim'])

    for n in range(N_TRIALS):
        prompt = dataset[n]
        if prepend_bos:
            # work out how to setup prompts for llama
            raise NotImplementedError
        activations_td = gather_activations(prompt, all_activation_layers, model, tokenizer)
        if split_attention:
            raise NotImplementedError
        
        for activation_type in activation_types:
            stack = torch.vstack([activations_td[layer].output[0] for layer in model_config[activation_type]])
            if final_activations_only:
                stack = stack[:,-1,:]
            else:
                raise NotImplementedError
                # Different activation shapes could be an issue here!
        
            activation_storage[activation_type][n] = stack

        

