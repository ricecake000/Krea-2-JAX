"""Qwen3-VL conditioner prompt template, shared by the data loader and the
JAX text encoder (and matching scripts/cache_dataset.py's TextEncoder).

The official Qwen3VLConditioner feeds the encoder
    PREFIX + caption + SUFFIX
and keeps the hidden states from the first caption token onward, i.e. rows
[len(prefix_ids):] = caption tokens, then the SUFFIX tokens. Padding tokens
are masked as keys and get no RoPE position (HF derives positions from the
attention mask), so the valid rows do not depend on how much padding follows
them; we therefore lay every sample out as
    [prefix | caption (<= text_rows - len(suffix)) | suffix | pad ...]
with a static total length len(prefix_ids) + text_rows.

`encode_captions` truncates captions from the END (same as the tokenizer's
`truncation=True` on PREFIX + caption in cache_dataset.py) so that caption +
suffix fits in `text_rows` rows.
"""

import numpy as np

PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, "
    "size, texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PAD_TOKEN = "<|endoftext|>"
# hidden_states indices (HF numbering: 0 = embeddings) stacked for the DiT
SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
# the real Qwen3-VL-4B tokenizer: PREFIX is 34 tokens, SUFFIX 5
QWEN3_PREFIX_LEN = 34
QWEN3_SUFFIX_LEN = 5
DEFAULT_TEXT_ROWS = 256


class CaptionTemplate:
    """Tokenizes captions into fixed-length encoder inputs.

    tokenizer: a `tokenizers.Tokenizer` (tokenizer.json of the text encoder;
    the prefix/suffix/pad ids are derived from it, so a tiny test tokenizer
    that defines the same special tokens works too).
    text_rows: rows handed to the DiT (caption + suffix), static.
    """

    def __init__(self, tokenizer, text_rows=DEFAULT_TEXT_ROWS):
        self.tokenizer = tokenizer
        self.text_rows = int(text_rows)
        self.prefix_ids = list(tokenizer.encode(PREFIX).ids)
        self.suffix_ids = list(tokenizer.encode(SUFFIX).ids)
        self.pad_id = tokenizer.token_to_id(PAD_TOKEN)
        if self.pad_id is None:
            raise ValueError(f"tokenizer has no {PAD_TOKEN!r} token")
        if self.text_rows <= len(self.suffix_ids):
            raise ValueError(
                f"text_rows {self.text_rows} must exceed the suffix length "
                f"{len(self.suffix_ids)}")

    @property
    def prefix_len(self):
        return len(self.prefix_ids)

    @property
    def total_len(self):
        """Encoder input length: prefix + text_rows."""
        return len(self.prefix_ids) + self.text_rows

    @property
    def max_caption_tokens(self):
        return self.text_rows - len(self.suffix_ids)

    def encode_captions(self, captions):
        """captions: list of str -> (ids (B, total_len) int32,
        txt_len (B,) int32 = caption tokens + suffix tokens, i.e. the number
        of valid rows after the prefix). Captions longer than
        max_caption_tokens are truncated from the end."""
        encs = self.tokenizer.encode_batch(list(captions))
        B = len(encs)
        ids = np.full((B, self.total_len), self.pad_id, dtype=np.int32)
        txt_len = np.zeros((B,), dtype=np.int32)
        p = len(self.prefix_ids)
        for b, e in enumerate(encs):
            cap = list(e.ids)[:self.max_caption_tokens]
            row = self.prefix_ids + cap + self.suffix_ids
            ids[b, :len(row)] = row
            txt_len[b] = len(cap) + len(self.suffix_ids)
        return ids, txt_len

    def valid_mask(self, txt_len):
        """(B, text_rows) bool for the rows after the prefix."""
        txt_len = np.asarray(txt_len)
        return np.arange(self.text_rows)[None, :] < txt_len[:, None]
