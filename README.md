# Retrieval lab

`lab.ipynb` improves one ranking step by step on two datasets, measuring every step. `lab.py` holds the helpers.
Python 3.10 or newer (on Ubuntu: `sudo apt install python3-venv`). Your lab page shows these with your URL and token.

Without a GPU (laptop): downloads the data with every precomputed vector (about 1.3 GB) and the question encoder
(1.2 GB); do it before the workshop if you can. The first pip line installs the CPU build of torch, which skips
about 3 GB of GPU libraries on Linux.

```bash
git clone https://github.com/oskrim/retrieval-lab && cd retrieval-lab
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python lab.py fetch
.venv/bin/python lab.py connect URL TOKEN
.venv/bin/jupyter lab lab.ipynb
```

With a GPU: the same download; set `ENCODE_ON_GPU = True` in the notebook's first cell and it encodes the
late-interaction token vectors again itself (about 10 minutes on an RTX 3090).

```bash
git clone https://github.com/oskrim/retrieval-lab && cd retrieval-lab
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-gpu.txt
.venv/bin/python lab.py fetch
.venv/bin/python lab.py connect URL TOKEN
.venv/bin/jupyter lab lab.ipynb
```

On a remote machine (for example a Verda GPU box), start Jupyter with `.venv/bin/jupyter lab --no-browser lab.ipynb`,
then run `ssh -L 8888:localhost:8888 USER@MACHINE` on your laptop and open the `http://localhost:8888/...` link it printed.
