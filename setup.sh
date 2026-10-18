#!/usr/bin/env bash
# Sets up the lab in this folder: a virtualenv (.venv) with the packages, and the data.
#   ./setup.sh               notebook packages + data + late-interaction token vectors (about 1.4 GB in all)
#   ./setup.sh --no-tokens   without the token vectors (the late-interaction section then needs a GPU)
#   ./setup.sh --gpu         also the GPU packages, to encode the token vectors yourself (ENCODE_ON_GPU = True)
set -euo pipefail
cd "$(dirname "$0")"
tokens=--tokens gpu=0
for arg in "$@"; do
  case $arg in
    --no-tokens) tokens= ;;
    --gpu) gpu=1 ;;
    *) echo "usage: ./setup.sh [--gpu] [--no-tokens]" >&2; exit 2 ;;
  esac
done
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { echo "needs Python 3.10 or newer" >&2; exit 1; }
python3 -m venv .venv || { echo "could not create .venv; on Ubuntu: sudo apt install python3-venv" >&2; exit 1; }
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
if [ "$gpu" = 1 ]; then .venv/bin/pip install -q -r requirements-gpu.txt; fi
.venv/bin/python ladder.py fetch $tokens
echo "Done. Next: the Connect and Notebook commands on your lab page (the link from the slide)."
