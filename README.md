# Retrieval lab

You improve a search ranking step by step and measure every step. The data: FiQA (personal-finance forum questions and
answers) and CAIQ (security questionnaire rows). The agents part uses CRM questions.

| Part | What you do |
|---|---|
| [1. Set up](#1-set-up-a-gpu-box-on-verda) | Rent a GPU box on Verda; a startup script installs everything |
| [2. Notebook, sections 1–19](#2-notebook) | A fast intro: BM25, vectors, late interaction, rerankers |
| [3. Vespa track](#3-vespa-track) | The main hands-on: tune a ranking in your own Vespa and hand it in |
| [4. Agents](#4-agents) | Notebook sections 20–23, then `lab new agent` |

## What the words mean

| Word | Meaning |
|---|---|
| Retrieval | Find the documents that answer a question, best first. |
| Reranker | A slower model that reads the question and one document together and scores them. It reorders the best candidates. |
| nDCG@10 | The score of your top 10 documents: 0 to 1, higher is better. |
| Dev questions | Their answers ship with the lab. `lab eval` scores them on your machine. |
| Board questions | Their answers are hidden. `lab submit` scores them and puts you on the leaderboard. |
| Vespa | A search engine. It runs in Docker on your box. |

## 1. Set up a GPU box on Verda

1. Get the coupon code from the organizers' email.
2. On Verda, create an **on-demand** instance: **A100 80GB** or **RTX PRO 6000 96GB**. Never spot: Verda reclaims
   spot boxes. A smaller GPU does not fit the agent model.
3. Add your SSH key.
4. Paste this URL as the startup script:
   `https://raw.githubusercontent.com/oskrim/retrieval-lab/main/setup.sh`
5. Wait about 12–15 min. The script clones the lab to `~/retrieval-lab`, installs it, runs `lab setup` and starts
   Vespa (`lab vespa up`).
6. Log in: `ssh root@MACHINE` (MACHINE is the IP on the Verda page). The user is `root`.
7. Run `ls ~/.lab`. You see `setup.done` when the script is finished. If you see `setup.failed`, send
   `~/.lab/setup.log` to the organizers.

Open your lab page from the organizers' link and pick a nickname. The page shows the commands below with your URL and
token filled in.

In every new terminal, activate the lab once:

```bash
cd ~/retrieval-lab && source .venv/bin/activate
```

Then connect and start the agent model:

```bash
lab connect <URL> <TOKEN>   # copy this line from your lab page
lab serve                   # the agent model, in the background; about 4 min
lab status                  # the GPU, the agent model, your connection and the downloads
```

## 2. Notebook

Jupyter runs on the box. You open it in your laptop's browser through an SSH tunnel.

On the box (this terminal stays busy; open a second SSH session for `lab` commands):

```bash
jupyter lab --no-browser --allow-root lab.ipynb
```

On your laptop:

```bash
ssh -L 8888:127.0.0.1:8888 root@MACHINE
```

Then open the `http://127.0.0.1:8888/lab?token=...` link that Jupyter printed.

Sections 1–19 are a fast intro. Do not stay long on them: the Vespa track is the main hands-on.

## 3. Vespa track

`vespa/app/schemas` holds the rank profiles: one per notebook technique. `vespa/search.py` sends the queries.

```bash
lab new vespa                                 # writes vespa/search.py and vespa/NICK
lab eval vespa/search.py                      # nDCG@10 on the dev questions
lab eval vespa/search.py --worst 5            # ... and the 5 worst questions with your top 10
lab submit vespa/search.py -m "what changed"  # instant score on the board questions
lab board --watch
```

Read the top of `vespa/search.py` for things to try.

| You change | Then run | Time |
|---|---|---|
| A `query(...)` input in `vespa/search.py` (you can set them per dataset) | `lab eval vespa/search.py` | No deploy. Eval: about 6 s for hybrid, 25–45 s with maxsim or rerank. |
| A rank profile in `vespa/app/schemas/fiqa.sd` | `lab vespa deploy`, then eval | |
| A field type in `fiqa.sd` | `lab vespa up` (feeds the data again) | About 1.5 min. Eval refuses until you run it. |

Edit only `fiqa.sd`. `lab vespa deploy` makes `caiq.sd` from it.

`lab vespa status` shows the memory and whether a schema change needs `lab vespa up`. `lab vespa down` removes Vespa
and its data.

### Hand in

1. On GitHub, fork `oskrim/retrieval-lab`.
2. Open `vespa/NICK`. The line after the comment must be your lab nick only. Replace `your-nick` if you see it.
3. Commit and push a branch to your fork. Git asks for a password: give a GitHub token with write access, not your
   GitHub password. Or push from your laptop.

   ```bash
   git config --global user.name "Your Name" && git config --global user.email you@example.com
   git checkout -b my-entry
   git add vespa/ && git commit -m "my entry"
   git remote add fork https://github.com/YOUR-GITHUB-USER/retrieval-lab
   git push fork my-entry
   ```

4. Send the organizers your fork URL and branch name, through the form or the email they give.

## 4. Agents

Do notebook sections 20–23 first. `lab serve` must be running (`lab status` shows it).

```bash
lab new agent                           # a working agent.py for the CRM questions
lab eval agent.py
lab submit agent.py -m "what changed"
lab board agent
```

## Search without Vespa

`lab new search` writes a `search.py` that ranks in Python: hybrid candidates, then the reranker. Use `lab eval` and
`lab submit` on it as above.

## No GPU (laptop)

You need Python 3.10 or newer, git, and Docker for the Vespa track. macOS and Linux work; on Windows, use WSL2.

```bash
git clone https://github.com/oskrim/retrieval-lab && cd retrieval-lab
python3 -m venv .venv && source .venv/bin/activate   # Ubuntu: sudo apt install python3-venv if this fails
pip install torch --index-url https://download.pytorch.org/whl/cpu   # skips about 3 GB of GPU libraries
pip install -e .
lab setup
lab connect <URL> <TOKEN>
lab vespa up       # Vespa in Docker; about 3 min
jupyter lab lab.ipynb
```

On Linux, Docker must run without sudo: `sudo apt install docker.io && sudo usermod -aG docker $USER`, then log in again.

What is different on a laptop:

- The reranker's scores come from a precomputed cache. It covers the top candidates; a deeper candidate scores lowest.
- The agent runs on the lab's model, so you skip `lab serve`.
