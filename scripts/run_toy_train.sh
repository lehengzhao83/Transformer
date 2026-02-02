#!/usr/bin/env bash
set -e

python -m src.train --task copy --epochs 10
python -m src.train --task reverse --epochs 20
