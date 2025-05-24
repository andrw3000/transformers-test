import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from captum.attr import (
    LayerIntegratedGradients,
    LLMGradientAttribution,
    TextTemplateInput,
)
import logging
import matplotlib.pyplot as plt
import numpy as np

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


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
    model,
    tokenizer,
    template=None,
    values=None,
    max_new_tokens=5,
):
    """Applies Integrated Gradients for generated token attribution."""

    # Forward pass model to obtain output
    if values:
        prompt = template.format(**values)
    else:
        prompt = template
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    generated = generate_text(model, input_ids, attention_mask, max_new_tokens)
    generated_tokens = generated[0][input_ids.size(1) :]
    output = tokenizer.decode(generated_tokens)

    logger.info(f"\nInput prompt:\n{prompt}")
    logger.info(f"\nGenerated output:\n{output}")

    # Attribution algorithm
    lig = LayerIntegratedGradients(model, model.model.embed_tokens)
    llm_attr = LLMGradientAttribution(lig, tokenizer)

    inp = TextTemplateInput(
        template=template,
        values=values,
        baselines=baselines,
    )
    attr_res = llm_attr.attribute(inp, target=output)
    logger.info(f"\nAttribution result: {attr_res}")

    def clean_token_for_log(token):
        # Replace common special characters used by tokenizers
        replacements = {
            "Ġ": " ",  # Space character in GPT tokenizers
            "Ċ": "\n",  # Newline character
            "ĊĊ": "\n",  # Double newline
            "▁": " ",  # Space character in some tokenizers
            "␣": " ",  # Visible space marker
            "▂": "_",  # Underscore representation
            "█": "[PAD]",  # Padding token
        }
        for char, replacement in replacements.items():
            token = token.replace(char, replacement)
        return token


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
        default="The {property} of the {region} which contains the {landmark} and has {item} as a {feature} is",
    )
    parser.add_argument(
        "--inspection_values",
        type=dict,
        default={
            "property": "capital",
            "region": "country",
            "landmark": "Eiffel Tower",
            "item": "frogs legs",
            "feature": "national dish",
        },
    )
    parser.add_argument(
        "--baselines",
        type=dict,
        default={
            "property": ["capital", "name"],
            "region": ["country", "region"],
            "landmark": ["Eiffel Tower", "Brandenburg Gate"],
            "item": ["frogs legs", "schlager"],
            "feature": ["national dish", "popular music genre"],
        },
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=1, help="How many tokens to generate"
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.local_dir)
    model = AutoModelForCausalLM.from_pretrained(args.local_dir)

    # Example of using a template
    interpret_generation(
        model,
        tokenizer,
        template=args.template,
        values=args.inspection_values,
        baselines=args.baselines,
        max_new_tokens=args.max_new_tokens,
    )
