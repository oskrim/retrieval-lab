# Retrieval lab

```bash
git clone https://github.com/oskrim/retrieval-lab && cd retrieval-lab && ./setup.sh
.venv/bin/python ladder.py connect URL TOKEN      # the Connect command on your lab page
.venv/bin/jupyter lab ladder.ipynb
```

`ladder.ipynb` improves one ranking step by step on two datasets, measuring every step. `ladder.py` holds the helpers.

`./setup.sh` makes a virtualenv with the packages (Python 3.10 or newer) and downloads the data, about 1.4 GB with the
late-interaction token vectors; download it before the workshop if you can. `./setup.sh --no-tokens` skips the token
vectors; `./setup.sh --gpu` adds the packages to encode them yourself (set `ENCODE_ON_GPU = True` in the notebook).

On a remote machine (for example a Verda GPU box): `.venv/bin/jupyter lab --no-browser ladder.ipynb` there, then
`ssh -L 8888:localhost:8888 USER@MACHINE` on your laptop and open the `http://localhost:8888/...` link it printed.
