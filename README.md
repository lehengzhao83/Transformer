# Transformer

This repository contains a minimal, from-scratch Transformer for toy seq2seq tasks (`copy` / `reverse`), refactored to comply with a **strict** `pyright` configuration and an opinionated `ruff` rule set.

---

## Features

- Encoder–decoder Transformer (teacher forcing)
- Boolean attention masks (padding, causal, cross) with **True = allowed**
- Toy dataset for sanity checking behavior (`copy`, `reverse`)
- Greedy decoding for qualitative inspection
- CPU-first workflow (assignment-friendly); AMP enabled only on CUDA

---

## Project Layout

```

.
├─ pyproject.toml
├─ run_toy_train.sh
├─ src/
│  └─ transformer/
│     ├─ **init**.py
│     ├─ data.py
│     ├─ eval.py
│     ├─ masks.py
│     ├─ model.py
│     ├─ train.py
│     └─ py.typed
└─ tests/
├─ test_attention_equivalence.py
├─ test_masks.py
└─ test_shapes.py

````

> **Important:** all imports should use `transformer.*` (not `src.*`).

---

## Requirements

- Python **3.12**
- PyTorch (CPU or CUDA)

---

## Installation (recommended)

Create a virtual environment and install in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .
````

This ensures `python -m transformer.train ...` works everywhere (including `pytest`).

---

## Quick Start: Train Toy Tasks

### Option A — Use the provided script

```bash
bash run_toy_train.sh
```

By default this runs on CPU. To run on CUDA (if available):

```bash
DEVICE=cuda bash run_toy_train.sh
```

### Option B — Run manually

Copy task:

```bash
python -m transformer.train --device cpu --task copy --epochs 10
```

Reverse task:

```bash
python -m transformer.train --device cpu --task reverse --epochs 20
```

---

## Configuration Notes

### Attention mask semantics

All masks are boolean and follow:

* `True`  → **allowed**
* `False` → **masked out**

In attention, this corresponds to:

```python
scores = scores.masked_fill(~mask, finfo_min)
```

### Special tokens (default)

* `PAD = 1`
* `BOS = 2`
* `EOS = 3`

---

## Quality Gates (Teacher Requirements)

Run these from repo root:

```bash
ruff check .
ruff format .
pyright
pytest -q
```

If everything is correctly refactored, all commands should pass.

---

## Common Issues & Fixes

### 1) `ModuleNotFoundError: transformer`

You likely didn't install editable mode.

Fix:

```bash
pip install -e .
```

### 2) Tests still import `src.*`

Update imports in `tests/` to `transformer.*`.
Remove any `sys.path` hacks that force `src` imports unless you explicitly want that behavior.

### 3) `pyright` complains about unknown lambda

Avoid `lambda` in `DataLoader(collate_fn=...)`.
Use `functools.partial(pad_collate_fn, pad_idx=pad_idx)` instead.

---

## What’s Being Tested

* **`test_attention_equivalence.py`**: our `MultiHeadAttention` matches `torch.nn.MultiheadAttention` (no mask case), after weight alignment.
* **`test_masks.py`**: padding, causal, decoder self-mask, and cross-mask behave correctly.
* **`test_shapes.py`**: embedding, attention, and Transformer output shapes are correct.

---
