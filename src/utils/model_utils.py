import torch
import accelerate

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM


def load_gpt_model_and_tokeniser(model_name: str, device: str = 'cuda'):
    """
    Load a huggingface model and its tokeniser
    """
    accelerator = accelerate.Accelerator()
    if model_name == 'gpt2-xl':
        #  Load the model form huggingface
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        MODEL_CONFIG={"n_heads":model.config.n_head,
                      "n_layers":model.config.n_layer,
                      "resid_dim":model.config.n_embd,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'transformer.h.{layer}.attn.c_proj' for layer in range(model.config.n_layer)],
                      "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.config.n_layer)]}
        
    elif 'gpt-j' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(device)

        MODEL_CONFIG={"n_heads":model.config.n_head,
                      "n_layers":model.config.n_layer,
                      "resid_dim":model.config.n_embd,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'transformer.h.{layer}.attn.out_proj' for layer in range(model.config.n_layer)],
                      "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.config.n_layer)]}
        
    elif model_name == 'llama':
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    # accelerate it, to use multi-gpu for inference!
    model, tokenizer = accelerator.prepare(model, tokenizer)

    return model, tokenizer, MODEL_CONFIG
    
