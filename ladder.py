"""Helpers for ladder.ipynb: the data, BM25, score matrices, nDCG@10, plots, and the lab's reranking API.
Plain Python and numpy.

  python3 ladder.py fetch [--tokens]       downloads the data (and the late-interaction token vectors) into ./data
  python3 ladder.py connect URL TOKEN      saves the lab URL and your token (both on your lab page) in .lab.json
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import sys
import tarfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from array import array
from dataclasses import dataclass

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
RELEASE = "https://github.com/oskrim/retrieval-lab/releases/download/data-v7"
LATE_MODEL = "perplexity-ai/pplx-embed-v2-late-0.6b"
TOKEN = re.compile(r"[a-z0-9]+")
LAB = HERE / ".lab.json"


# ---- data

@dataclass
class Dataset:
    name: str
    ids: list[str]
    titles: list[str]
    texts: list[str]
    queries: dict[str, str]  # question id -> text
    qrels: dict[str, dict[str, int]]  # question id -> doc id -> relevance grade (1 or 2)
    hidden: dict[str, set[str]]  # question id -> doc ids the question may not retrieve (its own answer and copies)
    title_text: list[str] | None = None  # each document's title and text together (the IDF of every field); built once

    def __post_init__(self) -> None:
        if self.title_text is None:
            self.title_text = [f"{t} {x}" for t, x in zip(self.titles, self.texts)]
        self.row = {d: i for i, d in enumerate(self.ids)}
        self.qids = list(self.queries)  # the row order of every score matrix
        self.qrow = {q: i for i, q in enumerate(self.qids)}
        self.hidden_mask = np.zeros((len(self.qids), len(self.ids)), dtype=bool)
        for q, docs in self.hidden.items():
            self.hidden_mask[self.qrow[q], [self.row[d] for d in docs if d in self.row]] = True


def fetch(tokens: bool = False) -> pathlib.Path:
    """Downloads and unpacks the data once (about 140 MB compressed) into ./data; tokens: also the late-interaction
    token vectors (about 1.2 GB)."""
    root = DATA / "data-v7"
    parts = [("data-v7.tar.gz", root)] + ([("late-tokens.tar.gz", root / "fiqa" / "late_tokens")] if tokens else [])
    for name, done in parts:
        if not done.exists():
            DATA.mkdir(exist_ok=True)
            urllib.request.urlretrieve(f"{RELEASE}/{name}", DATA / name)
            with tarfile.open(DATA / name) as t:
                t.extractall(DATA, filter="data")
            (DATA / name).unlink()
    return root


def load(name: str, split: str = "test") -> Dataset:
    """caiq: security questionnaire answers of several vendors; fiqa: finance forum posts. split: dev or test."""
    root = fetch() / name
    rows = lambda f: [json.loads(line) for line in open(root / f)]  # noqa: E731
    docs, qs = rows("docs.jsonl"), [q for q in rows("queries.jsonl") if q["split"] == split]
    qrels: dict[str, dict[str, int]] = {}
    for r in rows("qrels.jsonl"):
        qrels.setdefault(r["query"], {})[r["doc"]] = r["grade"]
    return Dataset(name, [d["id"] for d in docs], [d["title"] for d in docs], [d["text"] for d in docs],
                   {q["id"]: q["text"] for q in qs}, {q["id"]: qrels[q["id"]] for q in qs},
                   {q["id"]: set(q["hidden"]) for q in qs})


def single(ds: Dataset, text: str, qid: str | None = None) -> Dataset:
    """The dataset with one question, text; with qid, it hides the documents question qid hides and keeps its grades."""
    key = qid or "question"
    return Dataset(ds.name, ds.ids, ds.titles, ds.texts, {key: text}, {key: ds.qrels.get(qid, {})},
                   {key: ds.hidden.get(qid, set())}, ds.title_text)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


# ---- BM25

class BM25:
    """BM25 over one field: score(q, d) = sum over query terms t of idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg)).
    idf_texts: the texts whose document frequencies give idf (default: texts), e.g. every field of a document together."""

    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75, idf_texts: list[str] | None = None) -> None:
        self.vocab: dict[str, int] = {}
        rows, cols, lengths = array("q"), array("q"), np.zeros(len(texts))
        for i, text in enumerate(texts):
            toks = tokenize(text)
            lengths[i] = len(toks)
            for t in toks:
                rows.append(self.vocab.setdefault(t, len(self.vocab)))
                cols.append(i)
        key = np.frombuffer(rows, dtype=np.int64) * len(texts) + np.frombuffer(cols, dtype=np.int64)
        uniq, tf = np.unique(key, return_counts=True)
        term, self.doc, self.tf = uniq // len(texts), (uniq % len(texts)).astype(np.int64), tf.astype(np.float64)
        self.start = np.searchsorted(term, np.arange(len(self.vocab) + 1))
        df, n = np.diff(self.start), len(texts)
        if idf_texts is not None:
            counts: dict[str, int] = {}
            for text in idf_texts:
                for t in set(tokenize(text)):
                    if t in self.vocab:
                        counts[t] = counts.get(t, 0) + 1
            df, n = np.array([counts.get(t, 0) for t in self.vocab]), len(idf_texts)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        avg = lengths.mean() if lengths.any() else 1.0
        self.norm = k1 * (1 - b + b * lengths / avg)
        self.k1, self.n = k1, len(texts)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n)
        for t in set(tokenize(query)):
            if t in self.vocab:
                k = self.vocab[t]
                d, tf = self.doc[self.start[k]:self.start[k + 1]], self.tf[self.start[k]:self.start[k + 1]]
                out[d] += self.idf[k] * tf * (self.k1 + 1) / (tf + self.norm[d])
        return out


_indexes: dict[tuple[int, int, float, float], tuple[list[str], list[str] | None, BM25, dict[int, str]]] = {}


def index(texts: list[str], k1: float = 1.2, b: float = 0.75, idf_texts: list[str] | None = None) -> tuple[BM25, dict[int, str]]:
    """The BM25 index of texts, built once per list, and its cache of padded token strings for phrase()."""
    key = (id(texts), id(idf_texts), k1, b)
    if key not in _indexes:
        _indexes[key] = (texts, idf_texts, BM25(texts, k1, b, idf_texts), {})  # holding the lists keeps their ids from being reused
    return _indexes[key][2], _indexes[key][3]


# ---- score matrices: one row per question (in ds.qids order), one column per document

def bm25(ds: Dataset, texts: list[str], k1: float = 1.2, b: float = 0.75, idf_texts: list[str] | None = None) -> np.ndarray:
    """BM25 of every question against texts (one per document); idf_texts as in BM25."""
    index_, _ = index(texts, k1, b, idf_texts)
    return np.stack([index_.scores(ds.queries[q]) for q in ds.qids]).astype(np.float32)


def phrase(ds: Dataset, texts: list[str]) -> np.ndarray:
    """Per question and document: the share of the question's word pairs (bigrams) that appear as adjacent words."""
    index_, padded = index(texts)
    out = np.zeros((len(ds.qids), len(texts)), dtype=np.float32)
    for r, q in enumerate(ds.qids):
        toks = tokenize(ds.queries[q])
        grams = [(a, b) for a, b in zip(toks, toks[1:]) if a in index_.vocab and b in index_.vocab]
        for a, b in grams:
            docs = lambda t: index_.doc[index_.start[index_.vocab[t]]:index_.start[index_.vocab[t] + 1]]  # noqa: E731
            for i in np.intersect1d(docs(a), docs(b)):
                if i not in padded:
                    padded[i] = " " + " ".join(tokenize(texts[i])) + " "
                out[r, i] += f" {a} {b} " in padded[i]
        out[r] /= max(len(toks) - 1, 1)
    return out


def vectors(ds: Dataset, window: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """pplx-embed vectors at unit length: the documents, and the questions in ds.qids order. window 512 gives documents
    longer than 512 tokens the vector of their first 512 tokens, as an embedder with a 512-token window would."""
    if window not in (512, 8192):
        raise ValueError("window is 512 or 8192")
    z = np.load(fetch() / ds.name / "vectors.npz")
    docs = z["docs"].copy()
    if window == 512:
        docs[z["long"]] = z["truncated"]
    order = [json.loads(line)["id"] for line in open(fetch() / ds.name / "queries.jsonl")]
    qrow = {q: i for i, q in enumerate(order)}
    unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)  # noqa: E731
    return unit(docs.astype(np.float32)), unit(z["queries"][[qrow[q] for q in ds.qids]].astype(np.float32))


@dataclass
class Tokens:
    """Late-interaction token vectors (pplx-embed-v2-late-0.6b, 128 dimensions): documents in ds.ids order as int8 with
    one scale per vector, questions as float16."""
    docs: np.ndarray  # every document's vectors, one after another
    scale: np.ndarray
    start: np.ndarray  # document i's vectors are rows start[i] to start[i + 1]
    queries: dict[str, np.ndarray]  # question id -> its vectors
    center: np.ndarray | None = None  # the mean document vector, for centered bits; computed on first use

    def doc(self, i: int) -> np.ndarray:
        a, b = self.start[i], self.start[i + 1]
        return self.docs[a:b].astype(np.float32) * self.scale[a:b, None].astype(np.float32)

    def mean(self) -> np.ndarray:
        if self.center is None:
            total = np.zeros(self.docs.shape[1])
            for a in range(0, len(self.docs), 1_000_000):
                total += (self.docs[a:a + 1_000_000].astype(np.float32) * self.scale[a:a + 1_000_000, None].astype(np.float32)).sum(0)
            self.center = (total / len(self.docs)).astype(np.float32)
        return self.center


def late_tokens(ds: Dataset, encode_on_gpu: bool = False) -> Tokens:
    """The token vectors of ds's documents and questions: downloaded, or encoded on your GPU (encode_on_gpu; needs
    `./setup.sh --gpu`), once; both end up in ./data in the same form."""
    path = fetch() / ds.name / "late_tokens"
    if not path.exists():
        if encode_on_gpu:
            encode_late_tokens(ds.name, path)
        else:
            fetch(tokens=True)
    z = {f.stem: np.load(f, mmap_mode="r") for f in path.glob("*.npy")}
    assert list(z["doc_ids"]) == ds.ids, "the token vectors belong to another version of the data"
    offs = z["query_offsets"]
    return Tokens(z["docs"], z["scale"], z["doc_offsets"],
                  {str(q): np.asarray(z["queries"][offs[k]:offs[k + 1]], dtype=np.float32) for k, q in enumerate(z["query_ids"])})


def encode_late_tokens(name: str, path: pathlib.Path) -> None:
    """Encodes every document ("<title>\\n<text>") and question of dataset name on the GPU and saves them as int8."""
    import torch  # only this path needs the GPU packages
    from sentence_transformers import MultiVectorEncoder

    model = MultiVectorEncoder(LATE_MODEL, device="cuda")
    rows = lambda f: [json.loads(line) for line in open(fetch() / name / f)]  # noqa: E731
    docs, qs = rows("docs.jsonl"), rows("queries.jsonl")

    def encode(texts: list[str], fn) -> tuple[np.ndarray, np.ndarray]:
        out = [None] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        batch: list[int] = []
        for i in order + [None]:  # batches of about 24,000 characters, shortest texts first
            if batch and (i is None or sum(len(texts[j]) for j in batch) + len(texts[i]) > 24_000):
                with torch.inference_mode():
                    for j, e in zip(batch, fn([texts[j] or " " for j in batch], batch_size=len(batch))):
                        out[j] = np.asarray(e.float().cpu(), dtype=np.float32)
                batch = []
            if i is not None:
                batch.append(i)
        return np.concatenate(out), np.cumsum([0] + [len(e) for e in out])

    d, doffs = encode([f"{x['title']}\n{x['text']}".strip() for x in docs], model.encode_document)
    q, qoffs = encode([x["text"] for x in qs], model.encode_query)
    scale = np.maximum(np.abs(d).max(1), 1e-12) / 127
    path.mkdir()
    for k, v in dict(docs=np.clip(np.round(d / scale[:, None]), -127, 127).astype(np.int8), scale=scale.astype(np.float16),
                     doc_offsets=doffs, doc_ids=np.array([x["id"] for x in docs]), queries=q.astype(np.float16),
                     query_offsets=qoffs, query_ids=np.array([x["id"] for x in qs])).items():
        np.save(path / f"{k}.npy", v)


def maxsim(ds: Dataset, tokens: Tokens, top: np.ndarray, binary: bool = False) -> np.ndarray:
    """Late-interaction scores for the documents in top (questions × k document indices): for each question vector its
    best match among the document's vectors, summed. binary: compare centered sign bits instead."""
    signs = lambda x: np.where(x - tokens.mean() >= 0, 1, -1).astype(np.float32)  # noqa: E731
    out = np.empty(top.shape, dtype=np.float32)
    for r, q in enumerate(ds.qids):
        qv = signs(tokens.queries[q]) if binary else tokens.queries[q]
        for c, i in enumerate(top[r]):
            dv = signs(tokens.doc(i)) if binary else tokens.doc(i)
            out[r, c] = (qv @ dv.T).max(1).sum()
    return out


def topk(ds: Dataset, scores: np.ndarray, k: int = 10) -> np.ndarray:
    """Per question, the indices of its k best documents, best first, its hidden documents removed."""
    s = np.where(ds.hidden_mask, -np.inf, scores)
    best = np.argpartition(-s, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, best, 1), axis=1, kind="stable")
    return np.take_along_axis(best, order, 1)


def candidates(ds: Dataset, firsts: list[np.ndarray], k: int) -> np.ndarray:
    """True for every document in the top k of any of the score matrices in firsts."""
    out = np.zeros(ds.hidden_mask.shape, dtype=bool)
    for f in firsts:
        np.put_along_axis(out, topk(ds, f, k), True, 1)
    return out


def rescore(ds: Dataset, first: np.ndarray, then: np.ndarray, k: int) -> np.ndarray:
    """The scores in then for the k best documents by first, and -inf for every other document."""
    return np.where(candidates(ds, [first], k), then, -np.inf)


def ndcg10(ranked: list[str], grades: dict[str, int]) -> float:
    dcg = sum((2 ** grades.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(grades.values(), reverse=True)[:10]
    return dcg / sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))


def bootstrap(x: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean of x (one value per question), by resampling the questions with replacement."""
    means = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate(ds: Dataset, scores: np.ndarray) -> dict[str, float]:
    """nDCG@10 per question."""
    best = topk(ds, scores)
    return {q: ndcg10([ds.ids[i] for i in best[r]], ds.qrels[q]) for r, q in enumerate(ds.qids)}


def evaluate_top(ds: Dataset, top: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """nDCG@10 per question of the documents in top (questions × k document indices) ordered by scores (questions × k)."""
    ranked = np.take_along_axis(top, np.argsort(-scores, axis=1, kind="stable"), 1)
    return {q: ndcg10([ds.ids[i] for i in ranked[r]], ds.qrels[q]) for r, q in enumerate(ds.qids)}


def expand(ds: Dataset, top: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """A full score matrix from scores for the documents in top, -inf for every other document."""
    out = np.full(ds.hidden_mask.shape, -np.inf, dtype=np.float32)
    np.put_along_axis(out, top, scores, 1)
    return out


def top(ds: Dataset, qid: str, scores: np.ndarray, k: int = 10) -> list[str]:
    """The k best documents for one question; scores is a score matrix."""
    return [ds.ids[i] for i in topk(ds, scores[[ds.qrow[qid]]], k)[0]]


# ---- learning to rank

def labels(ds: Dataset, top: np.ndarray) -> np.ndarray:
    """The relevance grade of each document in top (questions × k document indices), 0 when unjudged."""
    return np.array([[ds.qrels[q].get(ds.ids[i], 0) for i in top[r]] for r, q in enumerate(ds.qids)])


def ideal_dcg(ds: Dataset) -> np.ndarray:
    """Per question, the DCG@10 of its perfect ranking: the denominator of ndcg10."""
    return np.array([sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(sorted(ds.qrels[q].values(), reverse=True)[:10]))
                     for q in ds.qids])


def lambdarank(X: np.ndarray, y: np.ndarray, ideal: np.ndarray, steps: int = 300, lr: float = 0.05,
               l2: float = 1e-3) -> np.ndarray:
    """Weights w ≥ 0 for the linear score X @ w (X: questions × k × features, y: their grades), trained with LambdaRank:
    for each pair where one document is more relevant, push it up by how much swapping the two would change nDCG@10."""
    w, m, v = np.zeros(X.shape[-1]), np.zeros(X.shape[-1]), np.zeros(X.shape[-1])
    gain = (2.0 ** y - 1) / ideal[:, None]
    better = y[:, :, None] > y[:, None, :]
    for t in range(1, steps + 1):
        s = X @ w
        rank = np.argsort(np.argsort(-s, 1), 1)
        discount = np.where(rank < 10, 1 / np.log2(rank + 2), 0.0)
        swap = np.abs((gain[:, :, None] - gain[:, None, :]) * (discount[:, :, None] - discount[:, None, :]))
        lam = np.where(better, swap / (1 + np.exp(np.clip(s[:, :, None] - s[:, None, :], -30, 30))), 0.0)
        grad = -np.einsum("nk,nkf->f", lam.sum(2) - lam.sum(1), X) / len(X) + l2 * w
        m, v = 0.9 * m + 0.1 * grad, 0.999 * v + 0.001 * grad ** 2  # Adam
        w = np.maximum(0, w - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8))  # a signal never counts against a document
    return w


def cross_validate(ds: Dataset, X: np.ndarray, y: np.ndarray, fit, folds: int = 5) -> np.ndarray:
    """Scores (questions × k) where each question is scored by fit(X_train, y_train, ideal_train, X_test), trained on
    the other folds of questions."""
    fold = np.random.default_rng(0).permutation(len(X)) % folds
    ideal = ideal_dcg(ds)
    out = np.zeros(y.shape)
    for k in range(folds):
        tr, te = fold != k, fold == k
        out[te] = fit(X[tr], y[tr], ideal[tr], X[te])
    return out


# ---- the lab's API

def connect(url: str, token: str) -> str:
    """Saves the lab URL and your token; returns your nickname."""
    LAB.write_text(json.dumps(dict(url=url.rstrip("/"), token=token)))
    return call("/api/me")["nick"]


def call(path: str, body: dict | None = None) -> dict:
    if not LAB.exists():
        raise RuntimeError("run the Connect command from your lab page first")
    lab = json.loads(LAB.read_text())
    req = urllib.request.Request(lab["url"] + path, None if body is None else json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {lab['token']}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{e.code}: {e.read().decode(errors='replace')}") from None


def rerank(ds: Dataset, top: np.ndarray, model: str) -> np.ndarray:
    """Cross-encoder scores for the documents in top (questions × k document indices). model: "qwen3-reranker-8b" or
    "voyage-rerank-3"."""
    out = np.empty(top.shape, dtype=np.float32)

    def one(r: int) -> None:
        res = call("/api/rerank", dict(d=ds.name, q=ds.queries[ds.qids[r]], doc_ids=[ds.ids[i] for i in top[r]], model=model))
        out[r] = res["scores"]

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, range(len(ds.qids))))
    return out


def grade(ds: Dataset, qid: str, answer: str) -> dict:
    """The CAIQ rubric's verdict on each row of an answer to question qid."""
    return call("/api/grade", dict(d=ds.name, qid=qid, answer=answer))


def submit(ds: Dataset, search_with, label: str) -> dict:
    """Runs search_with(ds, question, k=10) (document rows, best first) on the leaderboard's hidden FiQA questions and
    sends the results; the lab scores mean nDCG@10 and keeps your best. label: a few words on what your search does."""
    qs = call("/api/board/questions")["questions"]
    first = search_with(ds, qs[0]["text"], k=10)  # builds the search's indexes once, before the threads start
    with ThreadPoolExecutor(8) as pool:
        rows = [first] + list(pool.map(lambda q: search_with(ds, q["text"], k=10), qs[1:]))
    return call("/api/board", dict(label=label, ranking={q["id"]: [ds.ids[i] for i in r[:10]] for q, r in zip(qs, rows)}))


# ---- plots and examples

def plot_ladder(runs: dict[str, dict[str, dict[str, float]]]):
    """One panel per dataset: each step's mean nDCG@10 with a 95% interval, and its change from the step before."""
    import matplotlib.pyplot as plt

    steps = list(dict.fromkeys(step for r in runs.values() for step in r))
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 1.1 + 0.5 * len(steps)), sharex=True, sharey=True,
                             squeeze=False)
    for ax, (name, r) in zip(axes[0], runs.items()):
        prev = None
        for i, step in enumerate(steps):
            if step not in r:
                continue
            x = np.array(list(r[step].values()))
            half = 1.96 * x.std(ddof=1) / math.sqrt(len(x))
            ax.errorbar(x.mean(), i, xerr=half, fmt="o", color="C0", capsize=3)
            delta = f" ({x.mean() - prev:+.3f})" if prev is not None else ""
            ax.annotate(f"{x.mean():.3f}{delta}", (x.mean(), i), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
            prev = x.mean()
        ax.set_title(f"{name} ({len(next(iter(r.values())))} questions)")
        ax.set_xlabel("nDCG@10 (95% interval)")
    axes[0][0].set_xlim(0, 1)
    axes[0][0].set_ylim(len(steps) - 0.5, -0.8)
    axes[0][0].set_yticks(range(len(steps)), steps)
    fig.tight_layout()
    return fig


def show(ds: Dataset, qid: str, runs: dict[str, np.ndarray], k: int = 5, chars: int = 90) -> None:
    """The question, then each run's top k with relevance grades (2/1 relevant, . not); runs hold score matrices."""
    print(f"{qid}: {ds.queries[qid]}\n")
    for name, scores in runs.items():
        ranked = top(ds, qid, scores, k)
        print(f"{name}  nDCG@10 {ndcg10(top(ds, qid, scores), ds.qrels[qid]):.3f}")
        for i, d in enumerate(ranked, 1):
            text = " ".join(f"{ds.titles[ds.row[d]]} {ds.texts[ds.row[d]]}".split())
            g = ds.qrels[qid].get(d, 0)
            print(f"  {i}. [{g or '.'}] {d:<14} {text[:chars]}{'…' if len(text) > chars else ''}")
        print()



if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "connect":
        print(f"connected as {connect(sys.argv[2], sys.argv[3])}")
    elif sys.argv[1:2] == ["fetch"] and sys.argv[2:] in ([], ["--tokens"]):
        print(f"data in {fetch(tokens=sys.argv[2:] == ['--tokens'])}")
    else:
        sys.exit(__doc__)
