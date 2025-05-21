import os
import argparse
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def download_and_save_model(model_name: str, local_dir: str) -> None:
    """Download and save a transformer model and tokenizer locally."""

    logger.info(f"Downloading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Save the model and tokenizer locally
    logger.info(f"Saving model and tokenizer to {local_dir}...")
    tokenizer.save_pretrained(local_dir)
    model.save_pretrained(local_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and save a transformer model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="Name of the model to download",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./local_llm",
        help="Local directory to save the model",
    )

    args = parser.parse_args()
    download_and_save_model(args.model_name, args.local_dir)
    logger.info("Download and save completed successfully.")
