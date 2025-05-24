import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from captum.attr import (
    LayerIntegratedGradients,
    LLMGradientAttribution,
    TextTokenInput,
)
import logging
import matplotlib.pyplot as plt
import numpy as np

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def clean_token(token):
    """Replace common formatting in model outputs."""

    replacements = {
        "Ġ": " ",
        "Ċ": "\n",
        "ĊĊ": "\n",
        "▁": " ",
        "␣": " ",
        "▂": "_",
        "█": "[PAD]",
    }
    for char, replacement in replacements.items():
        token = token.replace(char, replacement)
    return token


def generate_text(model, input_ids, attention_mask, max_new_tokens=20):
    """Generate text using a language model."""
    model.eval()
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return output


def forward_func(embeddings, attention_mask=None, target_token_idx=None):
    """Model forward function for Captum."""
    outputs = model(inputs_embeds=embeddings)

    # Get the logits for the target token (last position)
    logits = outputs.logits[:, -1, :]

    # If we have a specific target token, select its logit
    if target_token_idx is not None:
        logits = logits[:, target_token_idx]

    return logits


def interpret_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    template: str,
    max_new_tokens: int = 20,
) -> dict[str, float]:
    """Applies Integrated Gradients for generated token attribution."""

    # Forward pass model to obtain output
    encoded = tokenizer(template, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    generated = generate_text(model, input_ids, attention_mask, max_new_tokens)
    generated_tokens = generated[0][input_ids.size(1) :]
    output = tokenizer.decode(generated_tokens)

    logger.info(f"\nInput prompt:\n{template}")
    logger.info(f"\nGenerated output:\n{output}")

    # Attribution algorithm
    lig = LayerIntegratedGradients(model, model.model.embed_tokens)
    llm_attr = LLMGradientAttribution(lig, tokenizer)
    inp = TextTokenInput(template, tokenizer, skip_tokens=[1])
    attr_res = llm_attr.attribute(inp, target=output)
    logger.info(f"\nSequence attribution dictionary:\n{attr_res.seq_attr_dict}")

    attr_res.plot_seq_attr(show=True)

    # Clean tokens and return dictionary of results
    return {clean_token(token): v for token, v in attr_res.seq_attr_dict.items()}


def plot_results(seq_attr_dict: dict[str, float], save_plot: bool = True) -> None:
    """Plot the attributions as a heat map over the original input senetence."""

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpret transformer text generation using Integrated Gradients."
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./local_llm",
        help="Directory of local model and tokeniser",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="The capital of the country associated to the Eiffel Tour and frogs' legs is",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=1, help="How many tokens to generate"
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.local_dir)
    model = AutoModelForCausalLM.from_pretrained(args.local_dir)

    # Example of using a template
    seq_attr_dict = interpret_generation(
        model,
        tokenizer,
        template=args.template,
        max_new_tokens=args.max_new_tokens,
    )
