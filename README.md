# 🎉 Tokenizer: Your Text Chunking CLI! 🦸‍♂️

Welcome to **Tokenizer**, the ultimate tool for splitting your long text files into manageable chunks of tokens using the OpenAI tiktoken package. Whether you're dealing with colossal documents or just want to break down text into digestible parts, Tokenizer has got your back!

## 🚀 Features

- **Read** your epic text files with ease.
- **Split** them into chunks based on token counts—no more, no less.
- **Save** each chunk as a separate file with a snazzy prefix of your choice.
- All powered by the magical tokenization capabilities of **GPT-4**.

## 🛠️ Installation

First things first, clone the repository:

```sh
git clone https://github.com/joseph-crowley/tokenizer.git
cd tokenizer
```

Then, set up a virtual environment and install the dependencies:

```sh
python3 -m venv venv
source venv/bin/activate
pip install tiktoken
```

Boom! You're ready to roll.

## 🎩 Usage

Running Tokenizer is as easy as pie. Just use the `main.py` script with your input text file and optional arguments to customize your chunking experience:

```sh
python tokenizer/main.py path/to/your/textfile.txt --output_prefix="chunk_" --max_tokens=4096 --model="gpt-4o"
```

### Arguments

- **`input_file`**: Path to your input text file.
- **`--output_prefix`**: Prefix for the output chunk files (default: `"chunk_"`).
- **`--max_tokens`**: Maximum number of tokens per chunk (default: `4096`).
- **`--model`**: Model to use for tokenization (default: `"gpt-4o"`).

### Example

Let's say you have a gigantic `epic_story.txt` file and you want to split it into chunks of 4096 tokens, saved with the prefix `epic_chunk_`. Here's how you'd do it:

```sh
python tokenizer/main.py epic_story.txt --output_prefix="epic_chunk_" --max_tokens=4096 --model="gpt-4o"
```

This will create files like `epic_chunk_0.txt`, `epic_chunk_1.txt`, and so on. Easy peasy!

## 📚 Detailed Breakdown

### Directory Structure

```
.
├── LICENSE
├── README.md
├── tokenizer
│   ├── __init__.py
│   ├── tokenizer.py
│   └── main.py
└── chunks
```

- `LICENSE`: Contains the licensing information for the project.
- `README.md`: This documentation file.
- `tokenizer`: Directory containing the tokenization logic.
  - `__init__.py`: Initialization file for the tokenizer module.
  - `tokenizer.py`: Contains the core tokenization and encoding logic.
  - `main.py`: Script to run the tokenization process from the command line.
- `chunks`: Directory where the output chunk files are saved.

### tokenizer/tokenizer.py

This is where the magic happens. The `tokenizer.py` script is packed with functions to:

- **Encode** text into tokens.
- **Decode** tokens back into text.
- **Split** text into chunks based on token counts.
- **Save** those chunks into files.

It leverages the powerful `tiktoken` library to handle all the heavy lifting.

#### Core Classes and Functions

- **Class `Encoding`**: Handles the encoding and decoding of text, including special tokens.
  - `__init__`: Initializes the encoding object with specified parameters.
  - `encode_ordinary`: Encodes a string into tokens, ignoring special tokens.
  - `encode`: Encodes a string into tokens, with options for allowed and disallowed special tokens.
  - `encode_ordinary_batch`: Encodes a list of strings in parallel, ignoring special tokens.
  - `encode_batch`: Encodes a list of strings in parallel.
  - `encode_with_unstable`: Encodes a string into stable tokens and possible completion sequences.
  - `encode_single_token`: Encodes a single token.
  - `decode_bytes`: Decodes tokens into bytes.
  - `decode`: Decodes tokens into a string.
  - `decode_single_token_bytes`: Decodes a single token into bytes.
  - `decode_tokens_bytes`: Decodes tokens into bytes.
  - `decode_with_offsets`: Decodes tokens into a string with offsets.
  - `decode_batch`: Decodes a batch of tokens in parallel.
  - `decode_bytes_batch`: Decodes a batch of tokens into bytes.
  - `token_byte_values`: Returns all token byte values.
  - `eot_token`: Returns the end-of-text token value.
  - `special_tokens_set`: Returns the set of special tokens.
  - `n_vocab`: Returns the number of tokens in the vocabulary.
- **Function `raise_disallowed_special_token`**: Raises an exception for disallowed special tokens.
- **Function `_special_token_regex`**: Returns a regex pattern for special tokens.
- **Function `split_text_into_chunks`**: Splits a long text into smaller chunks based on the token limit.
- **Function `save_chunks_to_files`**: Saves each chunk of text to a separate file with a given prefix.

### tokenizer/main.py

The `main.py` script is your command-line buddy. It reads your input text file, splits it into chunks using the functions from `tokenizer.py`, and saves those chunks with your chosen prefix.

#### Core Functions

- **Function `read_file`**: Reads the content of a file and returns it as a string.
- **Function `process_text`**: Processes the text by splitting it into chunks and saving them to files.
- **Function `parse_arguments`**: Parses command-line arguments.
- **Function `main`**: Main function to handle the overall process.

## 👨‍💻 Development

Want to dive deeper or contribute? Awesome! Here’s how you can set up your development environment:

1. Fork the repository.
2. Clone it to your local machine.
3. Create a new branch for your feature or bugfix.
4. Make your changes, commit, and push to your branch.
5. Create a pull request and let’s make Tokenizer even better together!

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by [Joe Crowley](https://github.com/joseph-crowley)

---

Now go forth and tokenize! 🎉