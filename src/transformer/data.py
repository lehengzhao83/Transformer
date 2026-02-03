from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class SpecialTokens:
    PAD: int = 1
    BOS: int = 2
    EOS: int = 3


class ToySeq2SeqDataset(Dataset[tuple[Tensor, Tensor]]):
    """
    Toy seq2seq dataset for correctness / behavior validation:
    - copy: target equals source
    - reverse: target is reversed source

    Each item returns:
      src: LongTensor (S,) with [BOS] + content + [EOS]
      tgt: LongTensor (T,) with [BOS] + transformed(content) + [EOS]
    """

    def __init__(
        self,
        size: int,
        min_len: int,
        max_len: int,
        vocab_size: int,
        task: str = "copy",
        specials: SpecialTokens | None = None,
        seed: int = 42,
    ) -> None:
        if task not in {"copy", "reverse"}:
            msg = f"task must be one of {{'copy', 'reverse'}}, got {task!r}"
            raise ValueError(msg)

        if size <= 0:
            raise ValueError("size must be positive")
        if min_len <= 0:
            raise ValueError("min_len must be positive")
        if max_len < min_len:
            raise ValueError("max_len must be >= min_len")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        st = specials if specials is not None else SpecialTokens()

        # Ensure vocab can hold special tokens + at least one content token.
        # content token range: [EOS+1, vocab_size-1]
        if vocab_size <= st.EOS + 1:
            msg = "vocab_size too small for special tokens; require vocab_size > specials.EOS + 1"
            raise ValueError(msg)

        self.size = size
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.task = task
        self.specials = st
        self.rng = random.Random(seed)

        self.low_token = st.EOS + 1
        self.high_token = vocab_size - 1

    def __len__(self) -> int:
        return self.size

    def _sample_seq(self) -> list[int]:
        length = self.rng.randint(self.min_len, self.max_len)
        return [self.rng.randint(self.low_token, self.high_token) for _ in range(length)]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        # idx is unused because this dataset is synthetic; kept for Dataset API compatibility.
        _ = idx

        x = self._sample_seq()

        if self.task == "copy":
            y = x
        else:
            y = list(reversed(x))

        src = [self.specials.BOS, *x, self.specials.EOS]
        tgt = [self.specials.BOS, *y, self.specials.EOS]

        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def pad_collate_fn(batch: Sequence[tuple[Tensor, Tensor]], pad_idx: int = 1) -> tuple[Tensor, Tensor]:
    """
    batch: sequence of (src, tgt), each is 1D LongTensor
    returns:
      src_padded: (B, Smax) LongTensor
      tgt_padded: (B, Tmax) LongTensor
    """
    if len(batch) == 0:
        raise ValueError("batch must be non-empty")

    src_list, tgt_list = zip(*batch, strict=True)

    src_lens = [int(s.numel()) for s in src_list]
    tgt_lens = [int(t.numel()) for t in tgt_list]

    bsz = len(batch)
    s_max = max(src_lens)
    t_max = max(tgt_lens)

    src_padded = torch.full((bsz, s_max), pad_idx, dtype=torch.long)
    tgt_padded = torch.full((bsz, t_max), pad_idx, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_list, tgt_list, strict=True)):
        if s.dim() != 1 or t.dim() != 1:
            raise ValueError("src/tgt must be 1D tensors")
        src_padded[i, : s.numel()] = s
        tgt_padded[i, : t.numel()] = t

    return src_padded, tgt_padded
