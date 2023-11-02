import torch, transformers, accelerate
import argparse

from src.utils.model_utils import load_gpt_model_and_tokeniser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neox-20b")

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokeniser(args.model_name)
    print("Model loaded.")