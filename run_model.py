import os
import argparse
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_model(model_dir: str) -> tuple:
    """Load and run a transformer model from a local directory.

    Args:
        model_dir: Path to the local model directory

    Returns:
        tuple: (model, tokenizer) pair
    """
    logger.info(f"Loading from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    # Run basic inference check
    test_input = "Hello, how are"
    inputs = tokenizer(test_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Test inference:\nInput: {test_input}\nOutput: {result}")

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and test a local transformer model"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./local_llm",
        help="Local directory containing the model",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode",
    )

    args = parser.parse_args()

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        model, tokenizer = run_model(args.model_dir)
        logger.info("Model type: %s", type(model).__name__)
        logger.info("Tokenizer type: %s", type(tokenizer).__name__)
        # Run interactive inference loop
        while True:
            try:
                user_input = input("\nEnter text (or 'q' to quit): ")
                if user_input.lower() == "q":
                    break
                inputs = tokenizer(user_input, return_tensors="pt")
                outputs = model.generate(
                    **inputs,
                    max_length=100,
                    num_return_sequences=1,
                    streamer=TextStreamer(tokenizer, skip_special_tokens=True),
                )
                # No need to decode since TextStreamer handles output
                result = ""  # Keep track of full response if needed
                print(f"\nOutput: {result}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error during inference: {e}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
