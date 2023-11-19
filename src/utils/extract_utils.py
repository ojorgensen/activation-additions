import os, re, json, gc

from typing import List, Dict

import torch
import einops
import numpy as np
import pandas as pd
from baukit import TraceDict
from tqdm import tqdm

from src.utils.memory_utils import print_cpu_memory, print_gpu_memory

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
    with TraceDict(model, layers=layers, retain_input=False, retain_output=True) as td:                
        model(**inputs) # batch_size x n_tokens x activation_dim, only want last token prediction

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
        DEBUG: bool=False,
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
        final_activations_only: whether to only collect the final activations
    Returns:
        activation_storage: dictionary of activations, with keys corresponding to activation types
            This gives a dictionary with keys corresponding to each prompt. The value of each of these
            is a tensor of activations, with shape (n_layers n_tokens hidden_size). If final_activations_only
            is True, then the shape is (n_layers 1 hidden_size).

    """
    def split_activations_by_head(activations, model_config):
        """
        Splits the output of the attention layer into heads
        TODO: understand how this works!
        """
        new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
        activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
        return activations

    # TODO: work out if we need to treat inputs to llama models differently
    is_llama = 'llama' in model_config['name_or_path']
    # prepend_bos = not is_llama
    activation_storage = {}

    # Get all layer activations we want
    all_activation_layers = []
    for activation_type in activation_types:
        all_activation_layers += model_config[activation_type]
        activation_storage[activation_type] = {}

    for n in tqdm(range(N_TRIALS), desc="Gathering activations"):
        torch.cuda.empty_cache()
        prompt = dataset[n]
        activations_td = gather_activations(prompt, all_activation_layers, model, tokenizer)
        if split_attention:
            raise NotImplementedError
        
        for activation_type in activation_types:
            stack = torch.stack([activations_td[layer].output[0].to(torch.device('cpu')) for layer in model_config[activation_type]]).cpu()
            # Shape should be (n_layers, n_tokens, hidden_size)
            # Removes batch dimension if it exists
            if len(stack.shape) == 4:
                stack = einops.rearrange(stack, 'n_layers 1 n_tokens hidden_size -> n_layers n_tokens hidden_size')
            # print(stack.shape)
            if final_activations_only:
                stack = stack[:,-1,:]
                stack = einops.rearrange(stack, 'n_layers hidden_size -> n_layers 1 hidden_size')
        
            activation_storage[activation_type][n] = stack.detach().cpu()
            if DEBUG:
                print_cpu_memory()
                print_gpu_memory()

        activations_td.close()
    
    return activation_storage

def average_vectors(activation_storage, model_config):
    """
    Computes the average vector of activations across all layers for a given activation type.
    Assuming layer hook names TODO: change this
    TODO: fix a memory leak currently here!
    Arguments:
        activation_storage: dictionary of activations
        model: huggingface model
        tokenizer: huggingface tokenizer
        model_config: dictionary of model configuration
    Returns:
        average_vector: tensor of shape (n_tokens, hidden_size)
    """
    # Make a tensor of the average activation in each layer
    average_tensor = torch.zeros(model_config['n_layers'], model_config['resid_dim'])
    total_tokens = 0
    for layer in range(model_config['n_layers']):
        for point in range(len(activation_storage['layer_hook_names'])):
            n_tokens = len(activation_storage['layer_hook_names'][point][layer].shape[1])
            total_tokens += n_tokens
            for token in range(n_tokens):
                average_tensor[layer] += activation_storage['layer_hook_names'][point][layer][:,token,:]
        average_tensor[layer] /= total_tokens
    
    return average_tensor

