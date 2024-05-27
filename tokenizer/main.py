import argparse
import logging
from pathlib import Path
from typing import List

from tokenizer import split_text_into_chunks, save_chunks_to_files

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_file(file_path: str) -> str:
    """
    Reads the content of a file and returns it as a string.

    :param file_path: Path to the input text file.
    :return: Content of the file as a string.
    """
    if not Path(file_path).is_file():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Reading file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        logger.info(f"Successfully read file: {file_path}")
        return content
    except Exception as e:
        logger.exception(f"Error reading file {file_path}: {e}")
        raise

def process_text(input_file: str, output_prefix: str, max_tokens: int, model: str) -> None:
    """
    Process the text from the input file, splitting it into chunks and saving to output files.

    :param input_file: Path to the input text file.
    :param output_prefix: Prefix for the output chunk files.
    :param max_tokens: Maximum number of tokens per chunk.
    :param model: Model to use for tokenization.
    """
    text = read_file(input_file)
    chunks = split_text_into_chunks(text, max_tokens, model)
    save_chunks_to_files(chunks, output_prefix)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: Parsed arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(description="Split a long text file into chunks based on token count.")
    parser.add_argument("input_file", type=str, help="Path to the input text file.")
    parser.add_argument("--output_prefix", type=str, default="chunk_", help="Prefix for the output chunk files.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum number of tokens per chunk.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use for tokenization.")
    
    return parser.parse_args()

def main() -> None:
    """
    Main function to handle the overall process.
    """
    args = parse_arguments()
    try:
        process_text(args.input_file, args.output_prefix, args.max_tokens, args.model)
    except Exception as e:
        logger.error(f"Failed to process text: {e}")
        raise

if __name__ == "__main__":
    main()
