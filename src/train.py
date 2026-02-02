import argparse
import math
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .model import Transformer
from .data import ToySeq2SeqDataset, pad_collate_fn, SpecialTokens
from .eval import evaluate_loss, greedy_decode


def resolve_device(device_arg: str) -> torch.device:
    """
    device_arg: "cpu" | "cuda" | "auto"
    - cpu: always CPU
    - cuda: GPU if available, else CPU (and prints a warning)
    - auto: GPU if available else CPU
    """
    device_arg = device_arg.lower()
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[Warning] --device cuda requested but CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    pad_idx: int = 1,
    grad_clip: float = 1.0,
    use_amp: bool = True,
):
    model.train()

    amp_enabled = (use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    total_loss = 0.0
    total_tokens = 0

    for src, tgt in loader:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        # teacher forcing shift
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(src, tgt_in)  # (B, T-1, V)
            V = logits.size(-1)
            loss = criterion(logits.reshape(-1, V), tgt_out.reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if grad_clip is not None:
            clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # token-weighted average loss (ignore PAD tokens)
        non_pad = (tgt_out != pad_idx).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    avg = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg, 20))
    return avg, ppl


def main():
    parser = argparse.ArgumentParser()

    # ✅ Device control: default to CPU for course requirement
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu",
                        help="Choose device. Default is CPU (recommended for assignment).")

    parser.add_argument("--task", choices=["copy", "reverse"], default="copy")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--train_size", type=int, default=4000)
    parser.add_argument("--valid_size", type=int, default=800)
    parser.add_argument("--min_len", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=20)

    parser.add_argument("--vocab_size", type=int, default=200)   # includes special tokens
    parser.add_argument("--max_len_model", type=int, default=256)

    # ✅ Default d_model=128 (meets your teacher requirement)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--ffn_hidden", type=int, default=512)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--no_amp", action="store_true",
                        help="Disable AMP. AMP is automatically disabled on CPU anyway.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save_path", type=str, default="transformer_best.pt")
    args = parser.parse_args()

    # basic argument sanity check
    if args.d_model % args.n_head != 0:
        raise ValueError(f"d_model ({args.d_model}) must be divisible by n_head ({args.n_head}).")

    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    # ✅ AMP only makes sense on CUDA; on CPU it will be disabled automatically in train/eval
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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: pad_collate_fn(b, pad_idx),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: pad_collate_fn(b, pad_idx),
    )

    model = Transformer(
        src_vocab_size=args.vocab_size,
        tgt_vocab_size=args.vocab_size,
        max_len=args.max_len_model,
        d_model=args.d_model,
        ffn_hidden=args.ffn_hidden,
        n_head=args.n_head,
        n_layer=args.n_layer,
        device=device,          # positional encoding buffer uses this device
        dropout=args.dropout,
        pad_idx=pad_idx,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            pad_idx=pad_idx, grad_clip=args.grad_clip, use_amp=use_amp
        )
        va_loss, va_ppl = evaluate_loss(
            model, valid_loader, criterion, device, pad_idx=pad_idx, use_amp=use_amp
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
                model, src,
                max_len=tgt.size(1),
                bos_idx=specials.BOS,
                eos_idx=specials.EOS,
                pad_idx=pad_idx
            )
            print("  sample src[0]:", src[0].tolist())
            print("  sample tgt[0]:", tgt[0].tolist())
            print("  sample dec[0]:", decoded[0].tolist())

    print("Done.")


if __name__ == "__main__":
    main()
