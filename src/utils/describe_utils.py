from typing import List, Dict
from datetime import datetime

import torch as t
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.extract_utils import gather_activations_from_dataset

def average_cosine_sim(
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
    Compute the average cosine similarity between activations of a layer of a model,
    using the dataset to produce activations.
    """
    label_dict = {
       "attn_hook_names": "attention layer",
        "layer_hook_names": "residual stream",
        "mlp_hook_names": "mlp layer",
    }

    # Produce all activations

    activation_storage = gather_activations_from_dataset(
        dataset, 
        activation_types, 
        model, 
        tokenizer, 
        model_config, 
        N_TRIALS, 
        split_attention,
        final_activations_only,
        DEBUG,   
    )
    num_layers = model_config['n_layers']

    results = {activation_type:{} for activation_type in activation_types}


    for layer in range(num_layers):
      for activation_type in activation_types:
        # Get the exact activations we want
        X = t.concatenate([activation_storage[activation_type][i][layer] for i in range(N_TRIALS)])
        # Norm all activations
        normed_X = X / X.norm(dim=1)[:, None]
        # Ensure float32
        normed_X = normed_X.float()
        # Compute pairwise activations
        sim = t.mm(normed_X, normed_X.t())

        # Get the upper triangle (only want to compute each pair once)
        sim_values = sim[t.triu(t.ones_like(sim), diagonal=1) == 1]

        # Find the average
        average = t.mean(sim_values)
        results[activation_type][layer] = average.item()

    # Plotting the histogram
    for activation_type in activation_types:
        plt.plot(results[activation_type].values(), label=label_dict.get(activation_type, activation_type))

    print(results)
    sns.set()
    params = {'legend.fontsize': 'xx-large',
              'figure.figsize': (8, 5),
            'axes.labelsize': 'xx-large',
            'axes.titlesize':'xx-large',
            'xtick.labelsize':'xx-large',
            'ytick.labelsize':'xx-large'}
    plt.rcParams.update(params)
    sns.set_style("whitegrid")
    sns.set_context("paper")
    # Adding labels and title
    model_name = model_config['name_or_path']
    time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    plt.xlabel('Layer Index', fontsize=16)
    plt.ylabel('Average Cosine Similarity', fontsize=16)
    plt.title(f'{model_name}', fontsize=16)
    # In the legend, name each line manually
    plt.legend(fontsize=16)  # Display the legend
    plt.savefig(f'results/cosine-sims/{model_name}-{time}.pdf', format='pdf')
    plt.show()

