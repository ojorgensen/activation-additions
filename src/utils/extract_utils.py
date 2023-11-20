import os, re, json, gc

from typing import List, Dict

import torch
import einops
import numpy as np
import pandas as pd
from baukit import TraceDict
from tqdm import tqdm

from src.utils.memory_utils import print_cpu_memory, print_gpu_memory
from src.utils.prompt_utils import get_token_meta_labels, word_pairs_to_prompt_data

# Layer Activations
def gather_layer_activations(prompt_data, layers, model, tokenizer):
    """
    Collects activations for an ICL prompt 

    Parameters:
    prompt_data: dict containing
    layers: layer names to get activatons from
    model: huggingface model
    tokenizer: huggingface tokenizer

    Returns:
    td: tracedict with stored activations
    """   
    
    # Get sentence and token labels
    query = prompt_data['query_target']['input']
    _, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query)
    sentence = [prompt_string]

    inputs = tokenizer(sentence, return_tensors='pt').to(model.device)

    # Access Activations 
    with TraceDict(model, layers=layers, retain_input=False, retain_output=True) as td:                
        model(**inputs) # batch_size x n_tokens x vocab_size, only want last token prediction

    return td


def get_mean_layer_activations(dataset, model, model_config, tokenizer, n_icl_examples = 10, N_TRIALS = 100, shuffle_labels=False, prefixes=None, separators=None, filter_set=None):
    """
    Computes the average activations for each layer in the model, at the final predictive token.

    Parameters: 
    dataset: ICL dataset
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    n_icl_examples: Number of shots in each in-context prompt
    N_TRIALS: Number of in-context prompts to average over
    shuffle_labels: Whether to shuffle the ICL labels or not
    prefixes: ICL template prefixes
    separators: ICL template separators
    filter_set: whether to only include samples the model gets correct via ICL

    Returns:
    mean_activations: avg activation of each layer hidden state of the model taken across n_trials ICL prompts
    """
    n_test_examples = 1
    activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['resid_dim'])

    if filter_set is None:
        filter_set = np.arange(len(dataset['valid']))

    is_llama = 'llama' in model_config['name_or_path']
    prepend_bos = not is_llama

    for n in range(N_TRIALS):
        word_pairs = dataset['train'][np.random.choice(len(dataset['train']),n_icl_examples, replace=False)]
        word_pairs_test = dataset['valid'][np.random.choice(filter_set,n_test_examples, replace=False)]
        if prefixes is not None and separators is not None:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, 
                                                    shuffle_labels=shuffle_labels, prefixes=prefixes, separators=separators)
        else:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=shuffle_labels)
        activations_td = gather_layer_activations(prompt_data=prompt_data, 
                                                  layers = model_config['layer_hook_names'], 
                                                  model=model, 
                                                  tokenizer=tokenizer)
        
        stack_initial = torch.vstack([activations_td[layer].output[0] for layer in model_config['layer_hook_names']])
        stack_filtered = stack_initial[:,-1,:] #Last token 
        
        activation_storage[n] = stack_filtered

    mean_activations = activation_storage.mean(dim=0)
    return mean_activations






# Ole's code below
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
    Assuming layer hook names TODO: fix this
    """
    average_tensor = torch.zeros(model_config['n_layers'], model_config['resid_dim'])

    n_datapoints = len(activation_storage['layer_hook_names'])

    # Compute the total number of tokens
    total_tokens = sum([activation_storage['layer_hook_names'][point][0].shape[0] for point in range(n_datapoints)])

    # Sum and average across the concatenated tensors for each layer
    for layer in range(model_config['n_layers']):
        activations = [activation_storage['layer_hook_names'][point][layer] for point in range(len(activation_storage['layer_hook_names']))]
        average_tensor[layer] = torch.sum(torch.concatenate(activations), dim=0) / total_tokens
    
    return average_tensor

def create_steering_vector(
        model,
        tokenizer,
        model_config,
        dataset1: List[str],
        dataset2: List[str],
        activation_types: List[str], 
        split_attention: bool=False,
        final_activations_only: bool=False,
        DEBUG: bool=False,
):
    """
    Computes the steering vector between two datasets.
    """
    # Gather activations for each dataset
    activation_storage1 = gather_activations_from_dataset(dataset1, activation_types, model, tokenizer, model_config, len(dataset1), split_attention, final_activations_only, DEBUG)
    activation_storage2 = gather_activations_from_dataset(dataset2, activation_types, model, tokenizer, model_config, len(dataset2), split_attention, final_activations_only, DEBUG)

    # Compute average vectors for each dataset
    average_tensor1 = average_vectors(activation_storage1, model_config)
    average_tensor2 = average_vectors(activation_storage2, model_config)

    # Compute steering vector
    steering_vector = average_tensor1 - average_tensor2

    return steering_vector



def create_mc_unmc_steering_vector(
        model,
        tokenizer,
        model_config,
        dataset1: List[str],
        dataset2: List[str],
        activation_types: List[str], 
        split_attention: bool=False,
        final_activations_only: bool=False,
        DEBUG: bool=False,
):
    """
    Computes the mean-centred and unmean-centred steering vector between two datasets.
    """
    # Gather activations for each dataset
    activation_storage1 = gather_activations_from_dataset(dataset1, activation_types, model, tokenizer, model_config, len(dataset1), split_attention, final_activations_only, DEBUG)
    activation_storage2 = gather_activations_from_dataset(dataset2, activation_types, model, tokenizer, model_config, len(dataset2), split_attention, final_activations_only, DEBUG)

    # Compute average vectors for each dataset
    average_tensor1 = average_vectors(activation_storage1, model_config)
    average_tensor2 = average_vectors(activation_storage2, model_config)

    # Compute steering vector
    mc_steering_vector = average_tensor1 - average_tensor2

    return mc_steering_vector, average_tensor1