from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from functools import partial
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .data import SpecialTokens, ToySeq2SeqDataset, pad_collate_fn
from .eval import evaluate_loss, greedy_decode
from .model import Transformer


@dataclass(frozen=True, slots=True)
class TrainArgs:
    device: str
    task: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float | None

    train_size: int
    valid_size: int
    min_len: int
    max_len: int

    vocab_size: int
    max_len_model: int

    d_model: int
    ffn_hidden: int
    n_head: int
    n_layer: int
    dropout: float

    no_amp: bool
    seed: int
    save_path: str


def resolve_device(device_arg: str) -> torch.device:
    """
    device_arg: "cpu" | "cuda" | "auto"
    - cpu: always CPU
    - cuda: GPU if available, else CPU (and prints a warning)
    - auto: GPU if available else CPU
    """
    arg = device_arg.lower()
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[Warning] --device cuda requested but CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast_enabled(device: torch.device, use_amp: bool) -> bool:
    return bool(use_amp and device.type == "cuda")


def _autocast_context(device: torch.device, use_amp: bool):
    enabled = _autocast_enabled(device, use_amp)
    # Prefer torch.amp.autocast on newer PyTorch; fallback for older versions.
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def train_one_epoch(
    model: Transformer,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    pad_idx: int = 1,
    grad_clip: float | None = 1.0,
    use_amp: bool = True,
) -> tuple[float, float]:
    model.train()

    amp_enabled = _autocast_enabled(device, use_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    total_loss = 0.0
    total_tokens = 0

    for src, tgt in loader:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        if tgt.dim() != 2:
            msg = f"tgt must be rank-2 (B,T); got shape={tuple(tgt.shape)}"
            raise ValueError(msg)

        # teacher forcing shift
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, use_amp):
            logits = model(src, tgt_in)  # (B, T-1, V)
            if logits.dim() != 3:
                msg = f"logits must be rank-3 (B,T,V); got shape={tuple(logits.shape)}"
                raise ValueError(msg)
            v = int(logits.size(-1))
            loss = criterion(logits.reshape(-1, v), tgt_out.reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if grad_clip is not None:
            clip_grad_norm_(model.parameters(), float(grad_clip))

        scaler.step(optimizer)
        scaler.update()

        # token-weighted average loss (ignore PAD tokens)
        non_pad = int((tgt_out != pad_idx).sum().item())
        total_loss += float(loss.detach().cpu().item()) * non_pad
        total_tokens += non_pad

    avg = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg, 20.0))
    return float(avg), float(ppl)


def _parse_args() -> TrainArgs:
    p = argparse.ArgumentParser()

    # Device control: default to CPU for course requirement
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Choose device. Default is CPU (recommended for assignment).",
    )

    p.add_argument("--task", choices=["copy", "reverse"], default="copy")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--train_size", type=int, default=4000)
    p.add_argument("--valid_size", type=int, default=800)
    p.add_argument("--min_len", type=int, default=5)
    p.add_argument("--max_len", type=int, default=20)

    p.add_argument("--vocab_size", type=int, default=200)  # includes special tokens
    p.add_argument("--max_len_model", type=int, default=256)

    # Default d_model=128 (meets teacher requirement)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--ffn_hidden", type=int, default=512)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--no_amp", action="store_true", help="Disable AMP. AMP is disabled on CPU anyway.")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--save_path", type=str, default="transformer_best.pt")

    ns = p.parse_args()

    grad_clip: float | None = float(ns.grad_clip)
    if grad_clip <= 0.0:
        # allow user to disable by passing <=0
        grad_clip = None

    return TrainArgs(
        device=str(ns.device),
        task=str(ns.task),
        epochs=int(ns.epochs),
        batch_size=int(ns.batch_size),
        lr=float(ns.lr),
        weight_decay=float(ns.weight_decay),
        grad_clip=grad_clip,
        train_size=int(ns.train_size),
        valid_size=int(ns.valid_size),
        min_len=int(ns.min_len),
        max_len=int(ns.max_len),
        vocab_size=int(ns.vocab_size),
        max_len_model=int(ns.max_len_model),
        d_model=int(ns.d_model),
        ffn_hidden=int(ns.ffn_hidden),
        n_head=int(ns.n_head),
        n_layer=int(ns.n_layer),
        dropout=float(ns.dropout),
        no_amp=bool(ns.no_amp),
        seed=int(ns.seed),
        save_path=str(ns.save_path),
    )


def main() -> int:
    args = _parse_args()

    # basic argument sanity check
    if args.d_model % args.n_head != 0:
        raise ValueError(f"d_model ({args.d_model}) must be divisible by n_head ({args.n_head}).")
    if args.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    use_amp = (not args.no_amp) and (device.type == "cuda")

    print(f"[Info] Using device: {device.type}")
    if device.type == "cpu":
        print("[Info] Running on CPU as required. (AMP disabled on CPU)")

    specials = SpecialTokens(PAD=1, BOS=2, EOS=3)
    pad_idx = specials.PAD

    train_ds = ToySeq2SeqDataset(
        size=args.train_size,
        min_len=args.min_len,
        max_len=args.max_len,
        vocab_size=args.vocab_size,
        task=args.task,
        specials=specials,
        seed=args.seed,
    )
    valid_ds = ToySeq2SeqDataset(
        size=args.valid_size,
        min_len=args.min_len,
        max_len=args.max_len,
        vocab_size=args.vocab_size,
        task=args.task,
        specials=specials,
        seed=args.seed + 1,
    )

    # Avoid lambda for strict typing / ruff reportUnknownLambdaType
    collate = partial(pad_collate_fn, pad_idx=pad_idx)

    train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )

    model = Transformer(
        src_vocab_size=args.vocab_size,
        tgt_vocab_size=args.vocab_size,
        max_len=args.max_len_model,
        d_model=args.d_model,
        ffn_hidden=args.ffn_hidden,
        n_head=args.n_head,
        n_layer=args.n_layer,
        device=device,  # kept for backward compatibility; model itself does not rely on it now
        dropout=args.dropout,
        pad_idx=pad_idx,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            pad_idx=pad_idx,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
        )
        va_loss, va_ppl = evaluate_loss(
            model,
            valid_loader,
            criterion,
            device,
            pad_idx=pad_idx,
            use_amp=use_amp,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} ppl {tr_ppl:.2f} | "
            f"val loss {va_loss:.4f} ppl {va_ppl:.2f}"
        )

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  saved: {args.save_path}")

        # small qualitative check: greedy decode one batch (optional)
        if epoch == 1 or epoch == args.epochs:
            src, tgt = next(iter(valid_loader))
            src = src[:2].to(device)
            tgt = tgt[:2].to(device)

            decoded = greedy_decode(
                model,
                src,
                max_len=int(tgt.size(1)),
                bos_idx=specials.BOS,
                eos_idx=specials.EOS,
                pad_idx=pad_idx,
            )
            print("  sample src[0]:", src[0].tolist())
            print("  sample tgt[0]:", tgt[0].tolist())
            print("  sample dec[0]:", decoded[0].tolist())

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
