import os, json
import torch, numpy as np
import argparse
from typing import List
from datetime import datetime

# Include prompt creation helper functions
from src.utils.prompt_utils import load_dataset
# from utils.intervention_utils import 
from src.utils.model_utils import load_gpt_model_and_tokeniser, set_seed
from src.utils.eval_utils import n_shot_eval, n_shot_eval_no_intervention
from src.utils.extract_utils import get_mean_layer_activations, gather_activations_from_dataset, average_vectors
from src.utils.dataset_utils import read_all_text_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='EleutherAI/gpt-j-6b')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='dataset_files')
    parser.add_argument('--save_path_root', help='File path to save mean activations to', type=str, required=False, default='results')
    parser.add_argument('--n_seeds', help='Number of seeds', type=int, required=False, default=5)
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", required=False, default=10)
    parser.add_argument('--n_trials', help="Number of in-context prompts to average over", required=False, default=100)
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)
    parser.add_argument('--device', help='Device to run on', required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--prefixes', help='Prompt template prefixes to be used', type=json.loads, required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    parser.add_argument('--separators', help='Prompt template separators to be used', type=json.loads, required=False, default={"input":"\n", "output":"\n\n", "instructions":""})    
    parser.add_argument('--layers', help='Layers to steer on', type=int, nargs='*', required=False, default=[])
    parser.add_argument('--training_set_size', help='Number of examples to use for training', type=int, required=False, default=300)
    parser.add_argument('--steering_coefficient', help='Coefficient to scale steering vector by', type=float, required=False, default=1.0)

    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    model_name = args.model_name
    root_data_dir = args.root_data_dir
    time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    save_path_root = f"{args.save_path_root}/{dataset_name}/{time}"
    n_seeds = args.n_seeds
    n_shots = args.n_shots
    n_trials = args.n_trials
    test_split = args.test_split
    device = args.device
    prefixes = args.prefixes
    separators = args.separators
    layers = args.layers
    training_set_size = args.training_set_size
    steering_coefficient = args.steering_coefficient
    

    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokeniser(model_name)
    seeds = np.random.choice(100000, size=n_seeds)

    # Create approximation of centre of model activations
    training_dataset = read_all_text_files("datasets/opentext_subset")
    if 'llama' in model_config['name_or_path']:
        training_dataset = [tokenizer.decode(tokenizer.encode(text)[:200])[4:] for text in training_dataset][:training_set_size]
    else:
        training_dataset = [tokenizer.decode(tokenizer.encode(text)[:200]) for text in training_dataset][:training_set_size]

    training_activations = gather_activations_from_dataset(
        training_dataset, ["layer_hook_names"], model, tokenizer, model_config, 
        len(training_dataset), False, False, False
    )

    centre_approximation = average_vectors(training_activations, model_config)
    
    for seed in seeds:
        set_seed(seed)

        # Load the dataset
        print("Loading Dataset")
        print("Current Dir:", os.getcwd())
        dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)

        print("Computing Mean Activations")
        dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, seed=seed)
        mean_activations = get_mean_layer_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer, 
                                                     n_icl_examples=n_shots, N_TRIALS=n_trials)
        
        mc_steering_vector = mean_activations - centre_approximation


        print("Saving mean layer activations")
        if not os.path.exists(save_path_root):
            os.makedirs(save_path_root)

        # Write args to file
        args.save_path_root = save_path_root # update for logging
        with open(f'{save_path_root}/mean_layer_activation_args.txt', 'w') as arg_file:
            json.dump(args.__dict__, arg_file, indent=2)

        torch.save(mean_activations, f'{save_path_root}/{dataset_name}_mean_layer_activations.pt')

        print("Evaluating Layer Avgs. Baseline")
        fs_results = n_shot_eval_no_intervention(dataset, n_shots, model, model_config, tokenizer)
        filter_set = np.where(np.array(fs_results['clean_rank_list']) == 0)[0]

        # Specify which layers to steer at
        if layers == []:
            layers = range(model_config['n_layers'])
        
        zs_res = {i:{} for i in layers}
        fss_res = {i:{} for i in layers}

        for i in layers:
            print("Evaluating original method")
            zs_res[i]['original'] = n_shot_eval(dataset, steering_coefficient * mean_activations[i].unsqueeze(0), i, 0, model, model_config, tokenizer, filter_set=filter_set)
            # Repeat with centred vector!
            print("Evaluating centred method")
            zs_res[i]['centred'] = n_shot_eval(dataset, steering_coefficient * mc_steering_vector[i].unsqueeze(0), i, 0, model, model_config, tokenizer, filter_set=filter_set)

        with open(f'{save_path_root}/mean_layer_intervention_zs_results_sweep_{seed}.json', 'w') as interv_zsres_file:
            json.dump(zs_res, interv_zsres_file, indent=2)