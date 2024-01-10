# Adapted from https://github.com/imoneoi/openchat/blob/master/ochat/scripts/hf_add_tokens.py
# License Apache-2.0

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def add_tokens_to_embedding(added_special_tokens, embedding):
    """Adds new tokens to an existing embedding layer.
    Args:
        added_special_tokens: A list of tokens to be added to the embedding layer.
        embedding: A torch.Tensor of shape (num_embeddings, embedding_dim) representing the
            existing embedding layer.

    Returns:
        A torch.Tensor of shape (num_embeddings + len(added_special_tokens), embedding_dim)
            containing the updated embedding layer with the new tokens added.
    """
    # Mean embedding, shape: [1, dim]
    new_token_embeddings = torch.mean(embedding.to(torch.float32), dim=0, keepdim=True).to(embedding.dtype)
    # Expand to [N, dim]
    new_token_embeddings = new_token_embeddings.expand(len(added_special_tokens), -1)

    return torch.cat([embedding, new_token_embeddings], dim=0)


def add_special_tokens(input_dir,
                       output_dir,
                       special_tokens,
                       low_memory=True, 
                       device='cpu'):
    """Adds special tokens to a pre-trained causal language model and its tokenizer.

    Args:
        input_dir: The directory containing the pre-trained model and tokenizer files.
        output_dir: The directory where the updated model and tokenizer will be saved.
        special_tokens: A list of tokens to be added to the model and tokenizer.
        low_memory: If True, uses a memory-efficient loading strategy for the model.
        device: The device to use for loading and processing (e.g., 'cpu' or 'cuda').
    """
    
    model = AutoModelForCausalLM(input_dir, 
                                 torch_dtype=torch.bfloat16,
                                 device_map=device,
                                 low_cpu_mem_usage=low_memory)
    tokenizer = AutoTokenizer(input_dir)

    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})


    model.model.embed_tokens.weight = torch.nn.Parameter(add_tokens_to_embedding(special_tokens, model.model.embed_tokens.weight), requires_grad=True)
    model.lm_head.weight = torch.nn.Parameter(add_tokens_to_embedding(special_tokens, model.lm_head.weight), requires_grad=True)

    model.config.vocab_size += len(special_tokens)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="model/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output_dir", default="model/TinyOpenChat-Untrained")
    parser.add_argument("--special_tokens", nargs='+')
    parser.add_argument("--low_memory", type=bool, default=True)
    parser.add_argument("--device", default="cpu")

    add_special_tokens(**vars(parser.parse_args()))