# Retrieval lab

`lab.ipynb` improves one ranking step by step on two datasets, measuring every step. `lab.py` holds the helpers, and
the `lab` command evaluates and submits your own search or agent. Everything runs on a GPU with 48 GB or more: the
question encoder, the reranker and the agent model (vLLM); without a GPU, see [No GPU](#no-gpu). Your lab page shows
these commands with your URL and token.

Create a GPU instance with 48 GB or more (Ubuntu, Python 3.10 or newer), log in with SSH and run:

```bash
git clone https://github.com/oskrim/retrieval-lab && cd retrieval-lab
python3 -m venv .venv && source .venv/bin/activate   # on Ubuntu, python3-venv first if this fails
pip install -e .
lab setup                # checks the GPU; downloads the data and models; installs vLLM; safe to re-run
lab connect URL TOKEN
lab serve                # starts the agent model in the background
jupyter lab --no-browser lab.ipynb
```

To open the notebook, run `ssh -L 8888:localhost:8888 USER@MACHINE` on your laptop and open the
`http://localhost:8888/...` link Jupyter printed.

Then, in a second SSH session (`source .venv/bin/activate` first):

```bash
lab new search                          # a working search.py: hybrid candidates, then the reranker
lab eval search.py                      # nDCG@10 on the dev questions, next to your best so far
lab eval search.py --worst 5            # ... and the 5 worst questions with your top 10
lab submit search.py -m "what changed"  # the leaderboard's questions; prints your score and rank
lab board --watch

lab new agent                           # a working agent.py for the CRM questions
lab eval agent.py
lab submit agent.py -m "what changed"
lab board agent
```

`lab status` shows the GPU, the agent model, your connection and the downloads; `lab serve --stop` stops the model.

## No GPU

The lab also runs on a laptop, with less: the reranker's scores come from a precomputed cache (a pair outside it ranks
lowest), and the agent runs through the lab's model instead of one on your machine. Install the CPU build of torch
first; it skips about 3 GB of GPU libraries on Linux.

```bash
git clone https://github.com/oskrim/retrieval-lab && cd retrieval-lab
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
lab setup                # the data, the question encoder and the reranker's scores
lab connect URL TOKEN
jupyter lab lab.ipynb
```

The notebook and every `lab` command above work, except `lab serve`.
