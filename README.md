# Retrieval lab

```bash
git clone https://github.com/oskrim/retrieval-lab && cd retrieval-lab
python3 -m venv .venv                          # Python 3.10 or newer (Ubuntu: sudo apt install python3-venv)
.venv/bin/pip install -r requirements.txt
.venv/bin/python ladder.py fetch --tokens      # the data and the late-interaction token vectors, about 1.4 GB
.venv/bin/python ladder.py connect URL TOKEN   # the URL and token on your lab page
.venv/bin/jupyter lab ladder.ipynb
```

`ladder.ipynb` improves one ranking step by step on two datasets, measuring every step. `ladder.py` holds the helpers.

Download the data before the workshop if you can. With a GPU you can encode the token vectors yourself instead:
`.venv/bin/pip install -r requirements-gpu.txt`, leave out `--tokens`, and set `ENCODE_ON_GPU = True` in the notebook.

On a remote machine (for example a Verda GPU box), start Jupyter with `.venv/bin/jupyter lab --no-browser ladder.ipynb`,
then run `ssh -L 8888:localhost:8888 USER@MACHINE` on your laptop and open the `http://localhost:8888/...` link it printed.
