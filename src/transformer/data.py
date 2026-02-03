from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class SpecialTokens:
    PAD: int = 1
    BOS: int = 2
    EOS: int = 3


DEFAULT_SPECIAL_TOKENS: Final[SpecialTokens] = SpecialTokens()


class ToySeq2SeqDataset(Dataset[tuple[Tensor, Tensor]]):
    """
    Toy seq2seq dataset for correctness / behavior validation:
    - copy: target equals source
    - reverse: target is reversed source
    """

    def __init__(
        self,
        size: int,
        min_len: int,
        max_len: int,
        vocab_size: int,
        task: str = "copy",
        specials: SpecialTokens = DEFAULT_SPECIAL_TOKENS,
        seed: int = 42,
    ) -> None:
        super().__init__()

        if task not in {"copy", "reverse"}:
            raise ValueError("task must be 'copy' or 'reverse'")
        if vocab_size <= specials.EOS + 1:
            raise ValueError("vocab_size too small for special tokens")

        self.size: int = size
        self.min_len: int = min_len
        self.max_len: int = max_len
        self.vocab_size: int = vocab_size
        self.task: str = task
        self.specials: SpecialTokens = specials
        self.rng = random.Random(seed)

        # tokens allowed for content (avoid PAD/BOS/EOS)
        self.low_token: Final[int] = specials.EOS + 1
        self.high_token: Final[int] = vocab_size - 1

    def __len__(self) -> int:
        return self.size

    def _sample_seq(self) -> list[int]:
        length = self.rng.randint(self.min_len, self.max_len)
        return [self.rng.randint(self.low_token, self.high_token) for _ in range(length)]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        _ = idx  # deterministic sampling uses RNG; idx unused
        x = self._sample_seq()
        y = x if self.task == "copy" else list(reversed(x))

        src = [self.specials.BOS, *x, self.specials.EOS]
        tgt = [self.specials.BOS, *y, self.specials.EOS]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def pad_collate_fn(batch: list[tuple[Tensor, Tensor]], pad_idx: int = 1) -> tuple[Tensor, Tensor]:
    """
    batch: list of (src, tgt), each is 1D LongTensor
    returns:
      src_padded: (B, Smax)
      tgt_padded: (B, Tmax)
    """
    if not batch:
        raise ValueError("batch is empty")

    src_list, tgt_list = zip(*batch, strict=True)

    src_lens = [int(s.numel()) for s in src_list]
    tgt_lens = [int(t.numel()) for t in tgt_list]

    bsz = len(batch)
    smax = max(src_lens)
    tmax = max(tgt_lens)

    src_padded = torch.full((bsz, smax), pad_idx, dtype=torch.long)
    tgt_padded = torch.full((bsz, tmax), pad_idx, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_list, tgt_list, strict=True)):
        src_padded[i, : s.numel()] = s
        tgt_padded[i, : t.numel()] = t

    return src_padded, tgt_padded
