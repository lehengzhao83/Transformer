
# Transformer

This repository contains a minimal Transformer encoder–decoder implementation in PyTorch, with **correctness verification** (unit tests) and a **basic CPU training loop** (d_model=128) that can run on a laptop/CPU-only machine.

The repo is designed as a clean starting point for future work (e.g., building larger models or migrating to a world-model codebase).

---

## Project Structure

```

.
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── pyrightconfig.json
├── src/
│   ├── **init**.py
│   ├── model.py
│   ├── masks.py
│   ├── data.py
│   ├── train.py
│   └── eval.py
└── tests/
├── conftest.py
├── test_shapes.py
├── test_masks.py
└── test_attention_equivalence.py

````

---

## Environment Setup (Linux)

All commands below should be run **in the repository root directory** (the folder where `README.md` is located).

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
````

If you see `(.venv)` at the beginning of your terminal prompt, the environment is active.

### 2) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

(Optional, recommended) Install development tools (lint + typing + tests):

```bash
pip install -r requirements-dev.txt
```

---

## Correctness Verification

The correctness verification is based on unit tests that check:

* **Shapes** of embedding, attention, encoder/decoder, and full Transformer outputs
* **Mask correctness** (padding mask + causal mask)
* **Numerical equivalence**: our `MultiHeadAttention` matches PyTorch’s official `nn.MultiheadAttention` when dropout=0 and no mask is used

Run all tests:

```bash
pytest -q
```

Expected result: you should see something like `... passed` and **no failures**.

---

## Code Quality (Ruff + Pyright)

This repo includes basic Python project management/config via `pyproject.toml` and type checking via `pyrightconfig.json`.

Run linting:

```bash
ruff check .
```

Run type checking:

```bash
pyright
```

(Optional) Auto-format code:

```bash
ruff format .
```

> Note: Type checking mode is set to `basic` to avoid excessive noise in educational settings, especially with PyTorch typing stubs.

---

## Basic Training Loop (CPU, d_model=128)

The training script uses:

* teacher forcing (shifted target input/output)
* `CrossEntropyLoss(ignore_index=PAD)`
* AdamW optimizer
* gradient clipping
* runs on **CPU by default** (even if CUDA is available) via `--device cpu`

### 1) Smoke test (1 epoch)

```bash
python -m src.train --device cpu --task copy --epochs 1 --d_model 128
```

You should see:

* `[Info] Using device: cpu`
* `Epoch 01 | train loss ... | val loss ...`
* the script completes with `Done.`

### 2) Behavior verification (loss should decrease)

Copy task:

```bash
python -m src.train --device cpu --task copy --epochs 10 --d_model 128
```

Reverse task:

```bash
python -m src.train --device cpu --task reverse --epochs 20 --d_model 128
```

Expected behavior:

* Train/val loss decreases over epochs (shows the model learns)
* Greedy decoding samples become closer to targets as training progresses

---

## Reproducibility Notes

* The script sets a manual random seed (`--seed`).
* CPU execution is enforced via `--device cpu`.
* Default model size is small (`d_model=128`) to ensure CPU feasibility.

---

