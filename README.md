Sign up at [Verda](https://console.verda.com/signin?signup) and add your SSH key:

```bash
mkdir -p ~/.ssh; [ -f ~/.ssh/id_verda ] || ssh-keygen -t ed25519 -N "" -q -f ~/.ssh/id_verda; cat ~/.ssh/id_verda.pub
```

Create a box in Verda (Ubuntu 24.04; a GPU box needs a 200 GB disk) and log in:

```bash
ssh -i ~/.ssh/id_verda -L 8888:127.0.0.1:8888 root@IP
```

Set it up and connect (5–20 min):

```bash
curl -fsSL https://lab.tryrealm.dev/setup | bash -s <TOKEN> && exec bash
```

The notebook is at http://127.0.0.1:8888/lab/tree/lab.ipynb.

Build a search in `search/`, then evaluate and submit:

```bash
lab eval search
lab submit search
```

`search/` holds everything that is graded: `search.py` (the query), `feed.py` (what Vespa holds: fields, chunks,
vectors), `analyze.py` (how a question is read: words, stopwords, stems), `synonyms.json` and `app/` (the Vespa
schema and services). `lab eval search` deploys an edited `app/` first (about 4 s). After an edit to `feed.py` or the
schema's document fields it asks you to run `lab vespa up`, which feeds again (about 2 minutes).

Every eval is saved as a run: `lab runs` lists them, `lab compare` shows what changed since the run before. An
agent run also keeps each question's trajectory: `lab trace crm 3` shows its tool calls, what they returned and
the answer. An agent answers differently from eval to eval, so evaluating an unchanged `agent.py` again keeps both:
`curious-parrot-60`, then `curious-parrot-60#2` (`lab trace crm 3 curious-parrot-60#2`; the name alone is its latest
eval).

Build an agent in `agent.py`:

```bash
lab eval agent
lab submit agent
```
