import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import AbstractSet, Collection, Literal, Union, List, Optional, Tuple

import regex
import tiktoken
from tiktoken import _tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Encoding:
    def __init__(self, name: str, pat_str: str, mergeable_ranks: dict[bytes, int], special_tokens: dict[str, int], explicit_n_vocab: Optional[int] = None):
        """
        Initializes an Encoding object.

        :param name: Name of the encoding.
        :param pat_str: Pattern string for the encoding.
        :param mergeable_ranks: Dictionary of mergeable ranks.
        :param special_tokens: Dictionary of special tokens.
        :param explicit_n_vocab: Optional explicit vocabulary size.
        """
        self.name = name
        self._pat_str = pat_str
        self._mergeable_ranks = mergeable_ranks
        self._special_tokens = special_tokens

        self.max_token_value = max(max(mergeable_ranks.values()), max(special_tokens.values(), default=0))
        if explicit_n_vocab:
            assert len(mergeable_ranks) + len(special_tokens) == explicit_n_vocab
            assert self.max_token_value == explicit_n_vocab - 1

        self._core_bpe = _tiktoken.CoreBPE(mergeable_ranks, special_tokens, pat_str)

    def __repr__(self) -> str:
        return f"<Encoding {self.name!r}>"

    def encode_ordinary(self, text: str) -> List[int]:
        """
        Encodes a string into tokens, ignoring special tokens.

        :param text: Input text.
        :return: List of token IDs.
        """
        try:
            return self._core_bpe.encode_ordinary(text)
        except UnicodeEncodeError:
            text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
            return self._core_bpe.encode_ordinary(text)

    def encode(self, text: str, allowed_special: Union[Literal["all"], AbstractSet[str]] = set(), disallowed_special: Union[Literal["all"], Collection[str]] = "all") -> List[int]:
        """
        Encodes a string into tokens.

        :param text: Input text.
        :param allowed_special: Special tokens allowed in the encoding.
        :param disallowed_special: Special tokens disallowed in the encoding.
        :return: List of token IDs.
        """
        if allowed_special == "all":
            allowed_special = self.special_tokens_set
        if disallowed_special == "all":
            disallowed_special = self.special_tokens_set - allowed_special
        if disallowed_special:
            disallowed_special = frozenset(disallowed_special)
            if match := _special_token_regex(disallowed_special).search(text):
                raise_disallowed_special_token(match.group())

        allowed_special = set(allowed_special)

        try:
            return self._core_bpe.encode(text, allowed_special)
        except UnicodeEncodeError:
            text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
            return self._core_bpe.encode(text, allowed_special)

    def encode_ordinary_batch(self, texts: List[str], num_threads: int = 8) -> List[List[int]]:
        """
        Encodes a list of strings into tokens, ignoring special tokens, in parallel.

        :param texts: List of input texts.
        :param num_threads: Number of threads to use for parallel encoding.
        :return: List of lists of token IDs.
        """
        with ThreadPoolExecutor(num_threads) as executor:
            return list(executor.map(self.encode_ordinary, texts))

    def encode_batch(self, texts: List[str], num_threads: int = 8, allowed_special: Union[Literal["all"], AbstractSet[str]] = set(), disallowed_special: Union[Literal["all"], Collection[str]] = "all") -> List[List[int]]:
        """
        Encodes a list of strings into tokens in parallel.

        :param texts: List of input texts.
        :param num_threads: Number of threads to use for parallel encoding.
        :param allowed_special: Special tokens allowed in the encoding.
        :param disallowed_special: Special tokens disallowed in the encoding.
        :return: List of lists of token IDs.
        """
        allowed_special = self.special_tokens_set if allowed_special == "all" else set(allowed_special)
        disallowed_special = self.special_tokens_set - allowed_special if disallowed_special == "all" else frozenset(disallowed_special)

        with ThreadPoolExecutor(num_threads) as executor:
            return list(executor.map(functools.partial(self.encode, allowed_special=allowed_special, disallowed_special=disallowed_special), texts))

    def encode_with_unstable(self, text: str, allowed_special: Union[Literal["all"], AbstractSet[str]] = set(), disallowed_special: Union[Literal["all"], Collection[str]] = "all") -> Tuple[List[int], List[List[int]]]:
        """
        Encodes a string into stable tokens and possible completion sequences.

        :param text: Input text.
        :param allowed_special: Special tokens allowed in the encoding.
        :param disallowed_special: Special tokens disallowed in the encoding.
        :return: Tuple of list of stable tokens and list of possible completion sequences.
        """
        allowed_special = self.special_tokens_set if allowed_special == "all" else set(allowed_special)
        disallowed_special = self.special_tokens_set - allowed_special if disallowed_special == "all" else frozenset(disallowed_special)
        if disallowed_special:
            if match := _special_token_regex(disallowed_special).search(text):
                raise_disallowed_special_token(match.group())

        return self._core_bpe.encode_with_unstable(text, allowed_special)

    def encode_single_token(self, text_or_bytes: Union[str, bytes]) -> int:
        """
        Encodes text corresponding to a single token to its token value.

        :param text_or_bytes: Input text or bytes.
        :return: Token value.
        """
        if isinstance(text_or_bytes, str):
            text_or_bytes = text_or_bytes.encode("utf-8")
        return self._core_bpe.encode_single_token(text_or_bytes)

    def decode_bytes(self, tokens: List[int]) -> bytes:
        """
        Decodes a list of tokens into bytes.

        :param tokens: List of token IDs.
        :return: Decoded bytes.
        """
        return self._core_bpe.decode_bytes(tokens)

    def decode(self, tokens: List[int], errors: str = "replace") -> str:
        """
        Decodes a list of tokens into a string.

        :param tokens: List of token IDs.
        :param errors: Error handling scheme.
        :return: Decoded string.
        """
        return self.decode_bytes(tokens).decode("utf-8", errors=errors)

    def decode_single_token_bytes(self, token: int) -> bytes:
        """
        Decodes a token into bytes.

        :param token: Token ID.
        :return: Decoded bytes.
        """
        return self._core_bpe.decode_single_token_bytes(token)

    def decode_tokens_bytes(self, tokens: List[int]) -> List[bytes]:
        """
        Decodes a list of tokens into a list of bytes.

        :param tokens: List of token IDs.
        :return: List of decoded bytes.
        """
        return [self.decode_single_token_bytes(token) for token in tokens]

    def decode_with_offsets(self, tokens: List[int]) -> Tuple[str, List[int]]:
        """
        Decodes a list of tokens into a string and a list of offsets.

        :param tokens: List of token IDs.
        :return: Tuple of decoded string and list of offsets.
        """
        token_bytes = self.decode_tokens_bytes(tokens)
        text_len = 0
        offsets = []
        for token in token_bytes:
            offsets.append(max(0, text_len - (0x80 <= token[0] < 0xC0)))
            text_len += sum(1 for c in token if not 0x80 <= c < 0xC0)
        text = b"".join(token_bytes).decode("utf-8", errors="strict")
        return text, offsets

    def decode_batch(self, batch: List[List[int]], errors: str = "replace", num_threads: int = 8) -> List[str]:
        """
        Decodes a batch (list of lists of tokens) into a list of strings.

        :param batch: List of lists of token IDs.
        :param errors: Error handling scheme.
        :param num_threads: Number of threads to use for parallel decoding.
        :return: List of decoded strings.
        """
        with ThreadPoolExecutor(num_threads) as executor:
            return list(executor.map(functools.partial(self.decode, errors=errors), batch))

    def decode_bytes_batch(self, batch: List[List[int]], num_threads: int = 8) -> List[bytes]:
        """
        Decodes a batch (list of lists of tokens) into a list of bytes.

        :param batch: List of lists of token IDs.
        :param num_threads: Number of threads to use for parallel decoding.
        :return: List of decoded bytes.
        """
        with ThreadPoolExecutor(num_threads) as executor:
            return list(executor.map(self.decode_bytes, batch))

    def token_byte_values(self) -> List[bytes]:
        """
        Returns the list of all token byte values.

        :return: List of token byte values.
        """
        return self._core_bpe.token_byte_values()

    @property
    def eot_token(self) -> int:
        """
        End of text token value.

        :return: End of text token value.
        """
        return self._special_tokens[""]

    @functools.cached_property
    def special_tokens_set(self) -> set[str]:
        """
        Returns the set of special tokens.

        :return: Set of special tokens.
        """
        return set(self._special_tokens.keys())

    @property
    def n_vocab(self) -> int:
        """
        Returns the number of tokens in the vocabulary.

        :return: Number of tokens in the vocabulary.
        """
        return self.max_token_value + 1

    def _encode_single_piece(self, text_or_bytes: Union[str, bytes]) -> List[int]:
        """
        Encodes text corresponding to bytes without a regex split.

        :param text_or_bytes: Input text or bytes.
        :return: List of token IDs.
        """
        if isinstance(text_or_bytes, str):
            text_or_bytes = text_or_bytes.encode("utf-8")
        return self._core_bpe.encode_single_piece(text_or_bytes)

    def _encode_only_native_bpe(self, text: str) -> List[int]:
        """
        Encodes a string into tokens, but do regex splitting in Python.

        :param text: Input text.
        :return: List of token IDs.
        """
        _unused_pat = regex.compile(self._pat_str)
        ret = []
        for piece in regex.findall(_unused_pat, text):
            ret.extend(self._core_bpe.encode_single_piece(piece))
        return ret

    def _encode_bytes(self, text: bytes) -> List[int]:
        """
        Encodes bytes into tokens.

        :param text: Input bytes.
        :return: List of token IDs.
        """
        return self._core_bpe._encode_bytes(text)

    def __getstate__(self) -> object:
        import tiktoken.registry
        if self is tiktoken.registry.ENCODINGS.get(self.name):
            return self.name
        return {
            "name": self.name,
            "pat_str": self._pat_str,
            "mergeable_ranks": self._mergeable_ranks,
            "special_tokens": self._special_tokens
        }

    def __setstate__(self, value: object) -> None:
        import tiktoken.registry
        if isinstance(value, str):
            self.__dict__ = tiktoken.registry.get_encoding(value).__dict__
            return
        value = dict(value)
        self.__init__(value.pop("name"), value.pop("pat_str"), value.pop("mergeable_ranks"), value.pop("special_tokens"))

def raise_disallowed_special_token(token: str) -> None:
    """
    Raises an exception for disallowed special tokens.

    :param token: Disallowed special token.
    """
    raise ValueError(f"Encountered text corresponding to disallowed special token: {token!r}. Please configure the `allowed_special` parameter for `tiktoken.Encoding.encode`/`encode_batch` to allow this token.")

@functools.cache
def _special_token_regex(tokens: AbstractSet[str]) -> "regex.Pattern[str]":
    """
    Returns a regex pattern for special tokens.

    :param tokens: Set of special tokens.
    :return: Compiled regex pattern.
    """
    return regex.compile(rf"""(?:{'|'.join(regex.escape(token) for token in tokens)})""")
    
def split_text_into_chunks(text: str, max_tokens: int = 4096, model: str = "gpt-4o") -> List[str]:
    """
    Splits a long text into smaller chunks based on the token limit.

    :param text: The input text to be split.
    :param max_tokens: Maximum number of tokens per chunk.
    :param model: Model to use for tokenization.
    :return: List of text chunks.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    chunks = [encoding.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
    return chunks

def save_chunks_to_files(chunks: List[str], prefix: str = "chunk_") -> None:
    """
    Saves each chunk of text to a separate file with a given prefix.

    :param chunks: List of text chunks.
    :param prefix: Prefix for the output files.
    """
    for idx, chunk in enumerate(chunks):
        filename = f"{prefix}{idx}.txt"
        try:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(chunk)
            logger.info(f"Saved chunk {idx} to {filename}")
        except IOError as e:
            logger.error(f"Failed to save chunk {idx} to {filename}: {e}")

