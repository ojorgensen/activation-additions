import torch
import accelerate
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM


def load_gpt_model_and_tokeniser(model_name: str):
    """
    Load a huggingface model and its tokeniser
    """
    access_token = open("llama-key.txt", "r").read()
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    if model_name == 'gpt2-xl':
        #  Load the model form huggingface
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)

        MODEL_CONFIG={"n_heads":model.config.n_head,
                      "n_layers":model.config.n_layer,
                      "resid_dim":model.config.n_embd,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'transformer.h.{layer}.attn.c_proj' for layer in range(model.config.n_layer)],
                      "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.config.n_layer)]}
        
    elif 'gpt-j' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map="auto")

        MODEL_CONFIG={"n_heads":model.config.n_head,
                      "n_layers":model.config.n_layer,
                      "resid_dim":model.config.n_embd,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'transformer.h.{layer}.attn.out_proj' for layer in range(model.config.n_layer)],
                      "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.config.n_layer)]}
    
    elif 'gpt-neox' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

        MODEL_CONFIG={"n_heads":model.config.num_attention_heads,
                      "n_layers":model.config.num_hidden_layers,
                      "resid_dim": model.config.hidden_size,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'gpt_neox.layers.{layer}.attention.dense' for layer in range(model.config.num_hidden_layers)],
                      "layer_hook_names":[f'gpt_neox.layers.{layer}' for layer in range(model.config.num_hidden_layers)]}
        
    elif 'llama' in model_name.lower():
        if '70b' in model_name.lower():
            # use quantization. requires `bitsandbytes` library
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            tokenizer = LlamaTokenizer.from_pretrained(model_name, token=access_token, device_map="auto")
            model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                    token=access_token,
                    device_map="auto",
            )
        else:
            if '7b' in model_name.lower():
                model_dtype = torch.float32
            else: #half precision for bigger llama models
                model_dtype = torch.float16
            tokenizer = LlamaTokenizer.from_pretrained(model_name, token=access_token, device_map="auto")
            model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype, token=access_token, device_map="auto")

        MODEL_CONFIG={"n_heads":model.config.num_attention_heads,
                      "n_layers":model.config.num_hidden_layers,
                      "resid_dim":model.config.hidden_size,
                      "name_or_path":model.config._name_or_path,
                      "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
                      "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)]}
    
    else:
        raise NotImplementedError
    
    # accelerate it, to use multi-gpu for inference!
    model, tokenizer = accelerator.prepare(model, tokenizer)

    return model, tokenizer, MODEL_CONFIG
    
