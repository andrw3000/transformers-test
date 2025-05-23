import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from captum.attr import IntegratedGradients
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def generate_text(model, tokenizer, input_ids, attention_mask, max_new_tokens=20):
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


def forward_func(input_ids, attention_mask=None):
    """Model forward function for Captum."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits


def interpret_generation(model, tokenizer, prompt, max_new_tokens=5):
    """Apply Integrated Gradients for generated token attribution."""
    # Tokenise input
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Generate tokens
    generated = generate_text(
        model, tokenizer, input_ids, attention_mask, max_new_tokens
    )
    generated_tokens = generated[0][input_ids.size(1) :]

    logger.info(f"\nPrompt: {prompt}")
    logger.info(f"Generated: {tokenizer.decode(generated_tokens)}")

    for i, target_token_id in enumerate(generated_tokens):
        # Full input up to and including the generated token
        current_ids = generated[0][: input_ids.size(1) + i + 1]
        current_ids = current_ids.unsqueeze(0)  # Add batch dim
        current_mask = torch.ones_like(current_ids)

        baseline_ids = torch.full_like(current_ids, tokenizer.pad_token_id)

        # Ensure type is LongTensor
        current_ids = current_ids.long()
        baseline_ids = baseline_ids.long()

        ig = IntegratedGradients(forward_func)

        attributions, delta = ig.attribute(
            inputs=current_ids,
            baselines=baseline_ids,
            additional_forward_args=(current_mask,),
            target=target_token_id.item(),
            return_convergence_delta=True,
        )

        tokens = tokenizer.convert_ids_to_tokens(current_ids[0])
        attributions_sum = attributions.sum(dim=-1).squeeze(0)
        attributions_norm = attributions_sum / torch.norm(attributions_sum)

        logger.info(
            f"\nAttributions for token: '{tokenizer.decode([target_token_id])}'"
        )
        for token, score in zip(tokens, attributions_norm):
            logger.info(f"{token}: {score.item():.4f}")
        logger.info(f"Convergence Delta: {delta.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpret transformer text generation using Captum."
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./local_llm",
        help="Directory of local model and tokenizer",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=5, help="How many tokens to generate"
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.local_dir)
    model = AutoModelForCausalLM.from_pretrained(args.local_dir)

    interpret_generation(model, tokenizer, args.prompt, args.max_new_tokens)
