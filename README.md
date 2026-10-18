# Retrieval lab

`ladder.ipynb` improves one ranking step by step on two datasets, measuring every step. `ladder.py` holds the helpers.
Python 3.10 or newer (on Ubuntu: `sudo apt install python3-venv`). Your lab page shows these with your URL and token.

Without a GPU (laptop): downloads the data and the late-interaction token vectors, about 1.4 GB; do it before the
workshop if you can.

```bash
git clone https://github.com/oskrim/retrieval-lab && cd retrieval-lab
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python ladder.py fetch --tokens
.venv/bin/python ladder.py connect URL TOKEN
.venv/bin/jupyter lab ladder.ipynb
```

With a GPU: downloads only the data (136 MB); set `ENCODE_ON_GPU = True` in the notebook's first cell and it encodes
the token vectors itself (about 10 minutes on an RTX 3090).

```bash
git clone https://github.com/oskrim/retrieval-lab && cd retrieval-lab
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-gpu.txt
.venv/bin/python ladder.py fetch
.venv/bin/python ladder.py connect URL TOKEN
.venv/bin/jupyter lab ladder.ipynb
```

On a remote machine (for example a Verda GPU box), start Jupyter with `.venv/bin/jupyter lab --no-browser ladder.ipynb`,
then run `ssh -L 8888:localhost:8888 USER@MACHINE` on your laptop and open the `http://localhost:8888/...` link it printed.
