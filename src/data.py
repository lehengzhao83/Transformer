import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class SpecialTokens:
    PAD: int = 1
    BOS: int = 2
    EOS: int = 3


class ToySeq2SeqDataset(Dataset):
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
        specials: SpecialTokens = SpecialTokens(),
        seed: int = 42,
    ):
        assert task in ["copy", "reverse"]
        assert vocab_size > specials.EOS + 1, "vocab_size too small for special tokens"
        self.size = size
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.task = task
        self.specials = specials
        self.rng = random.Random(seed)

        # tokens allowed for content (avoid PAD/BOS/EOS)
        self.low_token = specials.EOS + 1
        self.high_token = vocab_size - 1

    def __len__(self):
        return self.size

    def _sample_seq(self) -> List[int]:
        L = self.rng.randint(self.min_len, self.max_len)
        return [self.rng.randint(self.low_token, self.high_token) for _ in range(L)]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._sample_seq()

        if self.task == "copy":
            y = list(x)
        else:  # reverse
            y = list(reversed(x))

        # build src/tgt with BOS/EOS
        src = [self.specials.BOS] + x + [self.specials.EOS]
        tgt = [self.specials.BOS] + y + [self.specials.EOS]

        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def pad_collate_fn(batch, pad_idx: int = 1):
    """
    batch: list of (src, tgt), each is 1D LongTensor
    returns:
      src_padded: (B, Smax)
      tgt_padded: (B, Tmax)
    """
    src_list, tgt_list = zip(*batch)

    src_lens = [len(s) for s in src_list]
    tgt_lens = [len(t) for t in tgt_list]

    B = len(batch)
    Smax = max(src_lens)
    Tmax = max(tgt_lens)

    src_padded = torch.full((B, Smax), pad_idx, dtype=torch.long)
    tgt_padded = torch.full((B, Tmax), pad_idx, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_list, tgt_list)):
        src_padded[i, : s.numel()] = s
        tgt_padded[i, : t.numel()] = t

    return src_padded, tgt_padded
