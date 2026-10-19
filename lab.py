"""Helpers for lab.ipynb: the data, BM25, score matrices, nDCG@10, plots, and the lab's reranking API.
Plain Python and numpy.

  python3 lab.py fetch                  downloads the data, with every precomputed vector, into ./data, and the
                                        question encoder (embed) into the Hugging Face cache
  python3 lab.py connect URL TOKEN      saves the lab URL and your token (both on your lab page) in .lab.json
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
import sys
import tarfile
import threading
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from array import array
from dataclasses import dataclass

import numpy as np
import snowballstemmer

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
RELEASE = "https://github.com/oskrim/retrieval-lab/releases/download/data-v8"
LATE_MODEL = "perplexity-ai/pplx-embed-v2-late-0.6b"
QUERY_MODEL, QUERY_REVISION = "perplexity-ai/pplx-embed-v1-0.6b", "2c4d510dd4a732063c31a0f70193e35067b51fd8"
LAB = HERE / ".lab.json"


# ---- data

@dataclass
class Dataset:
    name: str
    ids: list[str]
    titles: list[str]
    bodies: list[str]
    queries: dict[str, str]  # question id -> text
    qrels: dict[str, dict[str, int]]  # question id -> doc id -> relevance grade (1 or 2)
    hidden: dict[str, set[str]]  # question id -> doc ids the question may not retrieve (its own answer and copies)
    title_body: list[str] | None = None  # each document's title and body together (the IDF of every field); built once

    def __post_init__(self) -> None:
        if self.title_body is None:
            self.title_body = [f"{t} {x}" for t, x in zip(self.titles, self.bodies)]
        self.row = {d: i for i, d in enumerate(self.ids)}
        self.qids = list(self.queries)  # the row order of every score matrix
        self.qrow = {q: i for i, q in enumerate(self.qids)}
        self.hidden_mask = np.zeros((len(self.qids), len(self.ids)), dtype=bool)
        for q, docs in self.hidden.items():
            self.hidden_mask[self.qrow[q], [self.row[d] for d in docs if d in self.row]] = True


def fetch() -> pathlib.Path:
    """Downloads and unpacks the data once into ./data: the documents, questions and grades, and every vector the
    notebook uses (dense, late-interaction tokens, the advanced sections' texts), about 1.3 GB."""
    root = DATA / "data-v8"
    if not root.exists():
        DATA.mkdir(exist_ok=True)
        urllib.request.urlretrieve(f"{RELEASE}/data-v8.tar.gz", DATA / "data-v8.tar.gz")
        with tarfile.open(DATA / "data-v8.tar.gz") as t:
            t.extractall(DATA, filter="data")
        (DATA / "data-v8.tar.gz").unlink()
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
    return Dataset(ds.name, ds.ids, ds.titles, ds.bodies, {key: text}, {key: ds.qrels.get(qid, {})},
                   {key: ds.hidden.get(qid, set())}, ds.title_body)


# ---- text: normalizing, tokens, terms

WORD = re.compile(r"[^\W_]+")  # runs of Unicode letters and digits; everything else separates words
STOPWORDS = frozenset("""a an and are as at be been but by can could did do does for from had has have how i if in into is
it its me my of on or our over should so than that the their them then there these they this those to was we were what
when where which while who why will with would you your""".split())
_STEMMER = snowballstemmer.stemmer("english")
_STEMS: dict[str, str] = {}  # every word is stemmed once


def normalize(text: str) -> str:
    """Lowercase, accents removed (é -> e), compatibility forms folded (ﬁ -> fi): Unicode NFKD, combining marks dropped."""
    if text.isascii():
        return text.lower()
    nfkd = unicodedata.normalize("NFKD", text)
    return nfkd.lower() if nfkd.isascii() else "".join(c for c in nfkd if not unicodedata.combining(c)).casefold()


def tokenize(text: str) -> list[str]:
    return WORD.findall(normalize(text))


def stem(word: str) -> str:
    """The word's Snowball (Porter 2) stem: encrypt, encrypted, encryption -> encrypt."""
    if word not in _STEMS:
        _STEMS[word] = _STEMMER.stemWord(word)
    return _STEMS[word]


def synonyms() -> list[dict]:
    """The synonym groups in synonyms.json: {phrases: [...]}, phrases that mean the same."""
    return json.loads((HERE / "synonyms.json").read_text())["groups"]


class Analyzer:
    """How text becomes the terms an index counts: its tokens, optionally without stopwords and stemmed. With synonym
    groups, a question's synonym phrases (found longest first, over stemmed tokens) become alternatives: each group
    scores as its best-matching alternative phrase (see Index.bm25)."""

    def __init__(self, stopwords: bool = False, stemming: bool = False, synonyms: list[dict] = ()) -> None:
        self.stopwords, self.stemming = stopwords, stemming
        self.groups = [[tokenize(p) for p in group["phrases"] if tokenize(p)] for group in synonyms]
        self.phrases: dict[tuple[str, ...], int] = {}
        for g, alternatives in enumerate(self.groups):
            for tokens in alternatives:
                self.phrases.setdefault(tuple(stem(t) for t in tokens), g)
        self.longest = max(map(len, self.phrases), default=0)
        self.starts = {k[0] for k in self.phrases}

    def word(self, token: str) -> str | None:
        """The term a token becomes, or None for a dropped stopword."""
        if self.stopwords and token in STOPWORDS:
            return None
        return stem(token) if self.stemming else token

    def terms(self, text: str) -> tuple[list[str], list[int]]:
        """The terms of text and their positions (the token's place in text, so dropped stopwords leave gaps)."""
        out = [(w, i) for i, w in enumerate(map(self.word, tokenize(text))) if w is not None]
        return [w for w, _ in out], [i for _, i in out]

    def query(self, text: str) -> tuple[list[str], list[tuple[list[str], list[list[tuple[int, str]]]]]]:
        """A question's terms outside synonym phrases, and for each synonym phrase in it: the phrase's own terms and
        the group's alternatives, each as (offset, term) pairs (a dropped stopword leaves its offset unused)."""
        tokens = tokenize(text)
        spans = self.concepts(tokens)
        inside = {i + k for _, i, n in spans for k in range(n)}
        words = [w for i, w in enumerate(map(self.word, tokens)) if w is not None and i not in inside]
        groups = []
        for g, i, n in spans:
            own = [w for w in map(self.word, tokens[i:i + n]) if w is not None]
            alternatives = [[(k, w) for k, w in enumerate(map(self.word, alt)) if w is not None] for alt in self.groups[g]]
            groups.append((own, [a for a in alternatives if a]))
        return words, groups

    def concepts(self, tokens: list[str]) -> list[tuple[int, int, int]]:
        """(group, position, length) of every synonym phrase in tokens: longest match first, left to right, no overlaps."""
        if not self.phrases:
            return []
        stems, found, i = [stem(t) for t in tokens], [], 0
        while i < len(stems):
            for n in range(min(self.longest, len(stems) - i), 0, -1) if stems[i] in self.starts else ():
                g = self.phrases.get(tuple(stems[i:i + n]))
                if g is not None:
                    found.append((g, i, n))
                    i += n
                    break
            else:
                i += 1
        return found


TOKENS = Analyzer()  # tokens as they are: no stopwords removed, no stemming, no synonyms


# ---- a positional inverted index

STRIDE = 1 << 32  # an occurrence's key is document * STRIDE + position


_tokenized: dict[int, tuple[list[str], dict[str, int], np.ndarray, np.ndarray]] = {}


def tokenized(texts: list[str]) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """texts as one array of word numbers, done once per list: (word -> number, the numbers of every text one after
    another, where each text starts)."""
    if id(texts) not in _tokenized:
        words: dict[str, int] = {}
        ids, start = array("q"), [0]
        for text in texts:
            ids.extend(words.setdefault(t, len(words)) for t in tokenize(text))
            start.append(len(ids))
        _tokenized[id(texts)] = (texts, words, np.frombuffer(ids, dtype=np.int64), np.array(start))  # holding texts keeps its id
    return _tokenized[id(texts)][1:]


class Index:
    """A positional inverted index of texts (one per document): for every term, the documents that contain it, how
    often (its postings), and at which positions. idf_texts: the texts whose document frequencies give idf (default:
    texts), e.g. every field of a document together."""

    def __init__(self, texts: list[str], analyzer: Analyzer = TOKENS, idf_texts: list[str] | None = None) -> None:
        self.analyzer, self.n = analyzer, len(texts)
        words, ids, start = tokenized(texts)
        self.vocab: dict[str, int] = {}
        to_term = np.array([-1 if w is None else self.vocab.setdefault(w, len(self.vocab))
                            for w in map(analyzer.word, words)] or [-1], dtype=np.int64)  # word number -> term, -1 if dropped
        doc = np.repeat(np.arange(self.n), np.diff(start))
        pos = np.arange(len(ids)) - start[doc]
        term = to_term[ids]
        kept = term >= 0
        term, doc, pos = term[kept], doc[kept], pos[kept]
        self.lengths = np.bincount(doc, minlength=self.n).astype(np.float64)
        order = np.argsort(term, kind="stable")  # occurrences come in document and position order already
        term, self.keys = term[order], doc[order] * STRIDE + pos[order]  # every occurrence: document * STRIDE + position
        self.key_start = np.searchsorted(term, np.arange(len(self.vocab) + 1))
        posting = term * self.n + doc[order]  # one per occurrence; equal neighbours are the same term in the same document
        first = np.flatnonzero(np.r_[True, posting[1:] != posting[:-1]]) if len(posting) else np.zeros(0, dtype=np.int64)
        self.doc, self.tf = posting[first] % self.n, np.diff(np.r_[first, len(posting)]).astype(np.float64)
        self.start = np.searchsorted(posting[first] // self.n, np.arange(len(self.vocab) + 1))
        df, n = np.diff(self.start), self.n
        if idf_texts is not None:  # document frequency over other texts, under the same analyzer
            other_words, other_ids, other_start = tokenized(idf_texts)
            other_term = np.array([self.vocab.get(w, -1) if w is not None else -1 for w in map(analyzer.word, other_words)]
                                  or [-1], dtype=np.int64)[other_ids]
            other_doc = np.repeat(np.arange(len(idf_texts)), np.diff(other_start))
            pairs = np.unique(other_term[other_term >= 0] * len(idf_texts) + other_doc[other_term >= 0])
            df, n = np.bincount(pairs // len(idf_texts), minlength=len(self.vocab)), len(idf_texts)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        self.avg = self.lengths.mean() if self.lengths.any() else 1.0

    def postings(self, term: str) -> tuple[np.ndarray, np.ndarray]:
        """The documents containing term and how often it occurs in each."""
        k = self.vocab.get(term)
        return (self.doc[self.start[k]:self.start[k + 1]], self.tf[self.start[k]:self.start[k + 1]]) if k is not None \
            else (np.zeros(0, dtype=np.int64), np.zeros(0))

    def occurrences(self, term: str) -> np.ndarray:
        """Every occurrence of term, as document * STRIDE + position, sorted."""
        k = self.vocab.get(term)
        return self.keys[self.key_start[k]:self.key_start[k + 1]] if k is not None else np.zeros(0, dtype=np.int64)

    def term_bm25(self, term: str, k1: float = 1.2, b: float = 0.75) -> np.ndarray:
        """One term's BM25 contribution in every document: idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg))."""
        out = np.zeros(self.n)
        docs, tf = self.postings(term)
        if len(docs):
            out[docs] = self.idf[self.vocab[term]] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.lengths[docs] / self.avg))
        return out

    def where(self, phrase: list[tuple[int, str]]) -> tuple[np.ndarray, np.ndarray]:
        """The documents in which the phrase's terms occur at its offsets (first term at p, the next at p + offset...),
        and how often."""
        (o0, t0), rest = phrase[0], phrase[1:]
        at = self.occurrences(t0)
        for o, t in rest:  # keep the starts whose next term sits the right distance after them
            following = self.occurrences(t)
            if not len(at) or not len(following):
                return np.zeros(0, dtype=np.int64), np.zeros(0)
            want = at + (o - o0)
            at = at[following[np.minimum(np.searchsorted(following, want), len(following) - 1)] == want]
        docs, tf = np.unique(at // STRIDE, return_counts=True)
        return docs, tf.astype(np.float64)

    def term_idf(self, term: str) -> float:
        """idf of term; a term no document contains gets the idf of the rarest possible term."""
        k = self.vocab.get(term)
        return float(self.idf[k]) if k is not None else float(np.log(1 + (self.n + 0.5) / 0.5))

    def bm25(self, query: str, k1: float = 1.2, b: float = 0.75) -> np.ndarray:
        """BM25 summed over the query's terms. A synonym phrase counts as its best alternative in each document (max
        across the alternatives). An alternative found as a phrase scores like one term with the question phrase's
        weight (the sum of its own terms' idf) and the alternative's count; the phrase's own terms also score as loose
        words, as plain BM25 would. So a synonym is worth as much as the question's own wording, never more."""
        words, groups = self.analyzer.query(query)
        out = sum((self.term_bm25(t, k1, b) for t in set(words)), np.zeros(self.n))
        for own, alternatives in groups:
            best = sum((self.term_bm25(t, k1, b) for t in set(own)), np.zeros(self.n))
            weight = sum(self.term_idf(t) for t in set(own))
            for phrase in alternatives:
                docs, tf = self.where(phrase)
                if len(docs):
                    norm = k1 * (1 - b + b * self.lengths[docs] / self.avg)
                    best[docs] = np.maximum(best[docs], weight * tf * (k1 + 1) / (tf + norm))
            out += best
        return out

    def phrase(self, query: str) -> np.ndarray:
        """Per document: the share of the query's neighbouring word pairs that occur in it the same distance apart.
        Pairs of two stopwords don't count. A dropped stopword keeps its place: "data at rest" needs rest two after data."""
        tokens = tokenize(query)
        words = [(i, self.analyzer.word(t), t in STOPWORDS) for i, t in enumerate(tokens)]
        words = [w for w in words if w[1] is not None]
        pairs = [(a, b) for a, b in zip(words, words[1:]) if not (a[2] and b[2])]
        out = np.zeros(self.n)
        for (i, a, _), (j, b, _) in pairs:
            first, second = self.occurrences(a) + (j - i), self.occurrences(b)
            few, many = (first, second) if len(first) <= len(second) else (second, first)
            if len(few) and len(many):  # look each of the rarer word's occurrences up in the other's sorted list
                hit = few[many[np.minimum(np.searchsorted(many, few), len(many) - 1)] == few]
                out[np.unique(hit // STRIDE)] += 1
        return out / max(len(pairs), 1)

    def matching(self, query: str) -> np.ndarray:
        """Per document: how many of the query's distinct terms it contains."""
        out = np.zeros(self.n)
        for t in set(self.analyzer.terms(query)[0]):
            out[self.postings(t)[0]] += 1
        return out

    def tfidf(self, query: str) -> np.ndarray:
        """score(q, d) = sum over q's terms t of tf(t, d) * idf(t)."""
        out = np.zeros(self.n)
        for t in set(self.analyzer.terms(query)[0]):
            docs, tf = self.postings(t)
            if len(docs):
                out[docs] += tf * self.idf[self.vocab[t]]
        return out

_indexes: dict[tuple[int, int, int], tuple[list[str], Analyzer, list[str] | None, Index]] = {}


def index(texts: list[str], analyzer: Analyzer = TOKENS, idf_texts: list[str] | None = None) -> Index:
    """The index of texts under analyzer, built once per (texts, analyzer, idf_texts)."""
    key = (id(texts), id(analyzer), id(idf_texts))
    if key not in _indexes:
        _indexes[key] = (texts, analyzer, idf_texts, Index(texts, analyzer, idf_texts))  # holding them keeps their ids unique
    return _indexes[key][3]


# ---- score matrices: one row per question (in ds.qids order), one column per document

def _rows(ds: Dataset, score) -> np.ndarray:
    return np.stack([score(ds.queries[q]) for q in ds.qids]).astype(np.float32)


def bm25(ds: Dataset, texts: list[str], k1: float = 1.2, b: float = 0.75, idf_texts: list[str] | None = None,
         analyzer: Analyzer = TOKENS) -> np.ndarray:
    """BM25 of every question against texts (one per document)."""
    ix = index(texts, analyzer, idf_texts)
    return _rows(ds, lambda q: ix.bm25(q, k1, b))


def phrase(ds: Dataset, texts: list[str], analyzer: Analyzer = TOKENS) -> np.ndarray:
    """Per question and document: the share of the question's word pairs found the same distance apart (Index.phrase)."""
    return _rows(ds, index(texts, analyzer).phrase)


def matching(ds: Dataset, texts: list[str], analyzer: Analyzer = TOKENS) -> np.ndarray:
    """Per question and document: how many of the question's distinct terms the document contains."""
    return _rows(ds, index(texts, analyzer).matching)


def tfidf(ds: Dataset, texts: list[str], analyzer: Analyzer = TOKENS, idf_texts: list[str] | None = None) -> np.ndarray:
    return _rows(ds, index(texts, analyzer, idf_texts).tfidf)


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
    """The token vectors of ds's documents and questions: the downloaded ones, or with encode_on_gpu your own, encoded
    once on your GPU (needs requirements-gpu.txt) into late_tokens_gpu next to them, in the same form."""
    path = fetch() / ds.name / ("late_tokens_gpu" if encode_on_gpu else "late_tokens")
    if not path.exists():
        encode_late_tokens(ds.name, path)
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
    """Per question, the indices of its k best documents, best first, its hidden documents removed. Blocks of 64
    questions run on parallel threads (numpy's partition releases the GIL)."""
    assert scores.shape == ds.hidden_mask.shape, "one row per question in ds.qids"
    out = np.empty((len(scores), k), dtype=np.int64)

    def block(rows: slice) -> None:
        s = np.where(ds.hidden_mask[rows], -np.inf, scores[rows])
        best = np.argpartition(-s, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(s, best, 1), axis=1, kind="stable")
        out[rows] = np.take_along_axis(best, order, 1)

    with ThreadPoolExecutor(min(8, os.cpu_count() or 1)) as pool:
        list(pool.map(block, [slice(i, i + 64) for i in range(0, len(scores), 64)]))
    return out


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
    """The k best documents for one question, its hidden documents removed; scores is a score matrix."""
    r = ds.qrow[qid]
    s = np.where(ds.hidden_mask[r], -np.inf, scores[r])
    return [ds.ids[i] for i in np.argsort(-s, kind="stable")[:k]]


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
    """Cross-encoder scores for the documents in top (questions × k document indices). model: "voyage-rerank-3"."""
    out = np.empty(top.shape, dtype=np.float32)

    def one(r: int) -> None:
        res = call("/api/rerank", dict(d=ds.name, q=ds.queries[ds.qids[r]], doc_ids=[ds.ids[i] for i in top[r]], model=model))
        out[r] = res["scores"]

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, range(len(ds.qids))))
    return out


def submit(ds: Dataset, search_with, label: str) -> dict:
    """Runs search_with(ds, question, k=10) (document rows, best first) on the leaderboard's FiQA questions (not in the data) and
    sends the results; the lab scores mean nDCG@10 and keeps your best. label: a few words on what your search does."""
    qs = call("/api/board/questions")["questions"]
    first = search_with(ds, qs[0]["text"], k=10)  # builds the search's indexes once, before the threads start
    with ThreadPoolExecutor(8) as pool:
        rows = [first] + list(pool.map(lambda q: search_with(ds, q["text"], k=10), qs[1:]))
    return call("/api/board", dict(label=label, ranking={q["id"]: [ds.ids[i] for i in r[:10]] for q, r in zip(qs, rows)}))


def submit_agent(answer_with, label: str) -> dict:
    """Runs answer_with(question) (a set of deal ids) on the agent leaderboard's CRM questions and sends the
    answers; the lab scores the share answered exactly and keeps your best. label: a few words on what your agent does."""
    qs = call("/api/agent-board/questions")["questions"]
    with ThreadPoolExecutor(8) as pool:
        found = list(pool.map(lambda q: sorted(answer_with(q["text"])), qs))
    return call("/api/agent-board", dict(label=label, answers={q["id"]: f for q, f in zip(qs, found)}))


# ---- plots and examples

def plot_progress(runs: dict[str, dict[str, dict[str, float]]]):
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
    print_progress(runs)
    return fig


def print_progress(runs: dict[str, dict[str, dict[str, float]]]) -> None:
    """The progress as a table: each step's mean nDCG@10 per dataset and its change from the step before."""
    steps = list(dict.fromkeys(step for r in runs.values() for step in r))
    width = max(map(len, steps)) + 2
    print(" " * width + "".join(f"{name:>10s}{'':9s}" for name in runs))
    prev: dict[str, float] = {}
    for step in steps:
        cells = []
        for name, r in runs.items():
            if step not in r:
                cells.append(" " * 19)
                continue
            mean = float(np.mean(list(r[step].values())))
            cells.append(f"{mean:10.3f}" + (f" ({mean - prev[name]:+.3f})" if name in prev else " " * 9))
            prev[name] = mean
        print(f"{step:{width}s}" + "".join(cells))


def example(ds: Dataset, before: dict[str, float], after: dict[str, float], broken: bool = False) -> str:
    """Prints what a step changes (nDCG@10 per question before and after) and returns a question it fixes: below 0.3
    before and above 0.7 after, the largest such gain (any question if none is). broken: a question it breaks instead."""
    change = {q: after[q] - before[q] for q in ds.qids}
    fixed = [q for q in ds.qids if before[q] < 0.3 and after[q] > 0.7]
    lost = [q for q in ds.qids if before[q] > 0.7 and after[q] < 0.3]
    print(f"{ds.name}: nDCG@10 {np.mean(list(before.values())):.3f} before, {np.mean(list(after.values())):.3f} after; better {sum(c > 0 for c in change.values())}, "
          f"worse {sum(c < 0 for c in change.values())} of {len(change)} questions; "
          f"fixed {len(fixed)}, broken {len(lost)} (from below 0.3 to above 0.7, or back)\n")
    if broken:
        return min(lost or ds.qids, key=change.get)
    return max(fixed or ds.qids, key=change.get)


def show(ds: Dataset, qid: str, runs: dict[str, np.ndarray], k: int = 5, chars: int = 90) -> None:
    """The question, then each run's top k with relevance grades (2/1 relevant, . not); runs hold score matrices."""
    print(f"{qid}: {ds.queries[qid]}\n")
    for name, scores in runs.items():
        ranked = top(ds, qid, scores, k)
        print(f"{name}  nDCG@10 {ndcg10(top(ds, qid, scores), ds.qrels[qid]):.3f}")
        for i, d in enumerate(ranked, 1):
            text = " ".join(f"{ds.titles[ds.row[d]]} {ds.bodies[ds.row[d]]}".split())
            g = ds.qrels[qid].get(d, 0)
            print(f"  {i}. [{g or '.'}] {d:<14} {text[:chars]}{'…' if len(text) > chars else ''}")
        print()


# ---- advanced: a fake CRM

CRM_TODAY = "2026-10-27"
INDUSTRIES = ("Financial services", "Insurance", "Healthcare", "Pharmaceuticals", "Retail", "E-commerce", "Logistics",
              "Manufacturing", "Automotive", "Energy", "Telecommunications", "Media", "Education", "Public sector",
              "Real estate", "Construction", "Hospitality", "Agriculture", "Software", "Legal services")
COUNTRIES = {"Finland": 3, "Sweden": 2, "Norway": 2, "Denmark": 1.5, "Iceland": 0.5, "Estonia": 1, "Germany": 2.5,
             "Austria": 1, "Switzerland": 1, "Netherlands": 1.5, "Belgium": 1, "France": 1.5, "Spain": 1, "Italy": 1,
             "Portugal": 0.5, "Poland": 1, "Czechia": 0.5, "Ireland": 0.5, "United Kingdom": 2, "United States": 2,
             "Canada": 1, "Brazil": 0.5, "Japan": 1, "Singapore": 0.5, "Australia": 1}  # country -> relative share of deals
STAGES = {"Prospecting": 0.1, "Qualification": 0.2, "Discovery": 0.3, "Proposal": 0.5, "Negotiation": 0.7,
          "Contract sent": 0.9, "Closed won": 1.0, "Closed lost": 0.0}  # stage -> probability of winning
OWNERS = ("Maria Virtanen", "Jonas Berg", "Anna Schmidt", "Lars Nilsen", "Sofia Rossi", "Mikko Laine", "Emma Johansson",
          "Lukas Weber", "Olivia Brown", "Pierre Martin", "Aino Korhonen", "Henrik Dahl", "Chloe Dubois", "Tomasz Nowak",
          "Sara Lindqvist")
PRODUCTS = ("Analytics Suite", "Data Platform", "Security Add-on", "Support Plus", "Mobile SDK", "Integration Hub",
            "AI Assistant", "Onboarding Services")
LEAD_SOURCES = ("Website", "Referral", "Partner", "Trade show", "Cold outreach", "Webinar", "Inbound call", "LinkedIn ad")
DESCRIPTIONS = {  # (field, value) -> words people use for it, so that value search has something to match
    **{("country", c): d for c, d in {
        "Finland": "Nordic country, Nordics, Scandinavia, Northern Europe, EU, Helsinki",
        "Sweden": "Nordic country, Nordics, Scandinavia, Northern Europe, EU, Stockholm",
        "Norway": "Nordic country, Nordics, Scandinavia, Northern Europe, Oslo",
        "Denmark": "Nordic country, Nordics, Scandinavia, Northern Europe, EU, Copenhagen",
        "Iceland": "Nordic country, Nordics, Northern Europe, Reykjavik",
        "Estonia": "Baltic country, Baltics, Northern Europe, EU, Tallinn",
        "Germany": "DACH, German-speaking, Central Europe, EU, Berlin, Munich",
        "Austria": "DACH, German-speaking, Central Europe, EU, Vienna",
        "Switzerland": "DACH, German-speaking, Central Europe, Zurich, Geneva",
        "Netherlands": "Benelux, Holland, Western Europe, EU, Amsterdam",
        "Belgium": "Benelux, Western Europe, EU, Brussels",
        "France": "Western Europe, EU, Paris", "Spain": "Southern Europe, Iberia, EU, Madrid",
        "Italy": "Southern Europe, EU, Milan, Rome", "Portugal": "Southern Europe, Iberia, EU, Lisbon",
        "Poland": "Central and Eastern Europe, CEE, EU, Warsaw", "Czechia": "Czech Republic, Central Europe, CEE, EU, Prague",
        "Ireland": "British Isles, Western Europe, EU, Dublin", "United Kingdom": "UK, Britain, England, British Isles, London",
        "United States": "USA, US, America, North America", "Canada": "North America, Toronto",
        "Brazil": "Latin America, South America, LATAM, Sao Paulo", "Japan": "Asia, APAC, Tokyo",
        "Singapore": "Asia, APAC, Southeast Asia", "Australia": "APAC, Oceania, Sydney"}.items()},
    **{("industry", i): d for i, d in {
        "Financial services": "banks, banking, fintech, payments, lending, asset management",
        "Insurance": "insurers, insurtech, underwriting, claims", "Healthcare": "hospitals, clinics, health care, medical",
        "Pharmaceuticals": "pharma, drug makers, life sciences, biotech", "Retail": "shops, stores, retailers, consumer goods",
        "E-commerce": "online stores, online retail, webshops", "Logistics": "shipping, freight, transport, supply chain",
        "Manufacturing": "factories, industrial, production", "Automotive": "car makers, cars, vehicles, auto industry",
        "Energy": "utilities, power, oil and gas, renewables", "Telecommunications": "telcos, telecom, mobile operators",
        "Media": "publishing, broadcasting, news, entertainment", "Education": "schools, universities, edtech",
        "Public sector": "government, municipalities, public administration, agencies",
        "Real estate": "property, proptech, landlords", "Construction": "builders, building, contractors, engineering",
        "Hospitality": "hotels, restaurants, travel, tourism", "Agriculture": "farming, agri, food production",
        "Software": "SaaS, tech companies, IT", "Legal services": "law firms, lawyers, legal"}.items()},
    **{("stage", st): d for st, d in {
        "Prospecting": "early stage, first contact, lead, top of the funnel, open",
        "Qualification": "early stage, qualifying the lead, open", "Discovery": "mid stage, needs analysis, open",
        "Proposal": "mid stage, quote sent, offer, open", "Negotiation": "late stage, agreeing price and terms, open",
        "Contract sent": "late stage, contract out for signature, about to sign, open",
        "Closed won": "won, signed, customer, closed", "Closed lost": "lost, dead, churned, closed"}.items()},
    **{("owner", o): f"salesperson, account owner, sales rep, {o.split()[0]}" for o in OWNERS},
    **{("product", pr): d for pr, d in {
        "Analytics Suite": "reporting, dashboards, BI", "Data Platform": "data warehouse, data lake, pipelines",
        "Security Add-on": "security, compliance, access control", "Support Plus": "premium support, service level",
        "Mobile SDK": "mobile apps, iOS, Android", "Integration Hub": "integrations, connectors, APIs",
        "AI Assistant": "AI, machine learning, chatbot, GenAI", "Onboarding Services": "implementation, training, setup"}.items()},
    **{("lead_source", ls): d for ls, d in {
        "Website": "web form, inbound, site", "Referral": "recommended by a customer, word of mouth",
        "Partner": "reseller, channel partner", "Trade show": "conference, expo, event, fair",
        "Cold outreach": "cold call, cold email, outbound", "Webinar": "online event, web seminar",
        "Inbound call": "phone, called us", "LinkedIn ad": "social media ad, LinkedIn, paid ads"}.items()},
}
FIELD_DESCRIPTIONS = {
    "amount_eur": "the value of the deal in euros: deal size, how big, biggest, largest, smallest, worth, value, price",
    "annual_revenue_eur": "the customer company's yearly revenue in euros: turnover, company size, how big the company is",
    "employees": "how many people work at the customer company: headcount, staff, company size",
    "probability": "the chance of winning the deal, 0 to 1, set by its stage: likelihood, win rate, how likely",
    "close_date": "the date the deal closed or is expected to close: closing, due, deadline, signing date",
    "last_activity_date": "the date of the latest call, email or meeting with the customer: last contact, last touch, "
                          "recent, gone quiet, stale, no contact since"}
_NAME = (("Aurora", "Boreal", "Cedar", "Delta", "Ember", "Fjord", "Granite", "Harbor", "Iris", "Juniper", "Kestrel",
          "Lumen", "Meridian", "Nordvik", "Orion", "Polar", "Quartz", "Riverstone", "Sable", "Tundra", "Umbra", "Vega",
          "Willow", "Zenith", "Alder", "Birch", "Coral", "Dune", "Elm", "Falcon"),
         ("Systems", "Group", "Labs", "Partners", "Holdings", "Solutions", "Industries", "Works", "Networks", "Logistics",
          "Health", "Capital", "Energy", "Foods", "Media", "Retail", "Motors", "Bank", "Insurance", "Studio"))
_NEEDS = ("wants to consolidate reporting across its regional offices", "needs to replace a legacy on-premise system",
          "wants to cut manual work in invoicing", "wants one view of every customer for the sales team",
          "must meet new audit requirements next year", "wants to speed up onboarding of new staff",
          "wants to automate support ticket routing", "needs better demand forecasting", "wants to move off spreadsheets",
          "needs an integration with SAP", "is tightening data security after a breach", "is expanding into two new markets")
_COMPETITORS = ("Pipewise", "Kestrel CRM", "Lumio", "Vantage", "Brightdesk")
_RIVALRY = ("Also evaluating {c}.", "Currently uses {c} and is unhappy with the pricing.", "{c} offered a 30% discount.",
            "Compared us with {c} in a bake-off.")
_RISKS = ("Budget freeze until the second quarter.", "Security review still pending.", "Our champion left the company.",
          "Legal is slow with the data processing agreement.", "Procurement wants three quotes.",
          "The CFO has not signed off yet.", "Strong executive sponsor, low risk.", "Pilot went well, users are happy.")


def crm(n: int = 600, seed: int = 0) -> list[dict]:
    """n fake deals, the same for the same seed. Revenue grows with headcount and the deal with the company; closed deals
    closed before CRM_TODAY, open deals close after it."""
    rng = np.random.default_rng(seed)
    today = np.datetime64(CRM_TODAY)
    names = rng.permutation([f"{a} {b}" for a in _NAME[0] for b in _NAME[1]])[:n]
    share = np.array(list(COUNTRIES.values()))
    stage_p = np.array([14, 12, 11, 10, 8, 5, 20, 20]) / 100  # share of deals in each stage, in STAGES order
    deals = []
    for i in range(n):
        stage = str(rng.choice(list(STAGES), p=stage_p))
        employees = int(np.clip(rng.lognormal(5.3, 1.7), 5, 150_000))
        amount = round(float(np.clip(2000 * employees ** 0.45 * rng.lognormal(0, 0.6), 3000, 2_000_000)), -2)
        closed = stage.startswith("Closed")
        close = today - int(rng.integers(1, 365)) if closed else today + int(rng.integers(1, 240))
        last = close - int(rng.integers(0, 14)) if closed else today - int(rng.exponential(30))
        notes = [f"{names[i]} {_NEEDS[rng.integers(len(_NEEDS))]}."]
        if rng.random() < 0.6:
            notes.append(_RIVALRY[rng.integers(len(_RIVALRY))].format(c=_COMPETITORS[rng.integers(len(_COMPETITORS))]))
        if rng.random() < 0.6:
            notes.append(_RISKS[rng.integers(len(_RISKS))])
        deals.append(dict(deal_id=f"D{i + 1:04d}", company=str(names[i]), industry=str(rng.choice(INDUSTRIES)),
                          country=str(rng.choice(list(COUNTRIES), p=share / share.sum())), stage=stage,
                          amount_eur=amount, annual_revenue_eur=round(employees * float(rng.lognormal(np.log(150_000), 0.5)), -4),
                          employees=employees, probability=STAGES[stage], close_date=str(close), last_activity_date=str(last),
                          owner=str(rng.choice(OWNERS)), product=str(rng.choice(PRODUCTS)),
                          lead_source=str(rng.choice(LEAD_SOURCES)), notes=" ".join(notes)))
    return deals



CATEGORIES = ("industry", "country", "stage", "owner", "product", "lead_source")



def embeddings() -> dict[str, np.ndarray]:
    """Precomputed pplx vectors (the question encoder, embed) of the texts the advanced sections embed, by text, from
    the data's embeddings.npz; embed makes the rest."""
    path = fetch() / "embeddings.npz"
    if not path.exists():
        return {}
    z = np.load(path)
    return dict(zip(z["texts"].tolist(), z["vectors"].astype(np.float32)))

_ENCODER: list = []  # the loaded question encoder, once
_ENCODER_LOCK = threading.Lock()


def embed(texts: list[str]) -> np.ndarray:
    """Unit pplx vectors of texts, made the way the data's question vectors were: pplx-embed-v1-0.6b (a 1.2 GB download
    on first use) on your GPU if you have one, else the CPU; mean pooling over at most 512 tokens, float32."""
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    import torch  # loaded on the first question only
    from sentence_transformers import SentenceTransformer

    with _ENCODER_LOCK:
        if not _ENCODER:
            full = SentenceTransformer(QUERY_MODEL, revision=QUERY_REVISION, trust_remote_code=True, device="cpu",
                                       model_kwargs={"torch_dtype": torch.float32})
            model = SentenceTransformer(modules=[full[0], full[1]], device="cuda" if torch.cuda.is_available() else "cpu")
            model.max_seq_length = 512  # the transformer and the mean pooling, without the model's int8 step
            _ENCODER.append(model.eval())
        v = _ENCODER[0].encode(list(texts), batch_size=16, convert_to_numpy=True, normalize_embeddings=True)
    assert v.ndim == 2 and len(v) == len(texts) and np.isfinite(v).all(), "pplx produced no usable vectors"
    return v.astype(np.float32)


def deal_text(d: dict) -> str:
    """A deal as a document: its notes, then its attributes in words, so that plain search can find them too."""
    return d["notes"] + " " + "; ".join(f"{f.replace('_', ' ')}: {d[f]}" for f in d if f not in ("deal_id", "company", "notes")) + "."


def documents(name: str) -> tuple[list[str], list[str], list[str]]:
    """(ids, titles, bodies) of the advanced sections' small datasets: "crm" (the deals), "crm-values" (every value of
    the category fields, described), "crm-fields" (the number and date fields, described), "tools" (tool_catalog())."""
    if name == "crm":
        deals = crm()
        return [d["deal_id"] for d in deals], [d["company"] for d in deals], [deal_text(d) for d in deals]
    if name == "crm-values":
        deals = crm()
        values = [(f, v) for f in CATEGORIES for v in sorted({d[f] for d in deals})]
        return [f"{f}={v}" for f, v in values], [f"{f.replace('_', ' ')}: {v}" for f, v in values], [DESCRIPTIONS[f, v] for f, v in values]
    if name == "crm-fields":
        return list(FIELD_DESCRIPTIONS), [f.replace("_", " ") for f in FIELD_DESCRIPTIONS], list(FIELD_DESCRIPTIONS.values())
    if name == "tools":
        fns = [t["function"] for t in tool_catalog()]
        return ([f["name"] for f in fns], [f["name"].replace("_", " ") for f in fns],
                [f"{f['description']} Parameters: {', '.join(f['parameters']['properties'])}." for f in fns])
    raise ValueError(f"no dataset {name!r}")

# ---- advanced: a catalog of tools

_CATALOG = """
crm_search_contacts | Find people in the CRM by name, email or company; returns their phone, email and role. | query
crm_create_contact | Add a new person to the CRM. | name, email, company, role
crm_update_deal_stage | Move a sales deal to another pipeline stage. | deal_id, stage
crm_log_call | Record a phone call with a customer contact: who, when and what was discussed. | contact_id, summary
crm_create_deal | Open a new sales opportunity for a company. | company, amount_eur, close_date
crm_get_account | Everything about one customer company: its contacts, open deals and recent activity. | account_id
create_calendar_event | Add an event to your own calendar. | title, start, end, attendees
schedule_meeting_room | Reserve a meeting room for a time slot. | room, start, end
send_meeting_invite | Invite more people to a calendar event that already exists. | event_id, emails
find_free_slot | Find times when all the given people are free. | attendees, duration_minutes
cancel_calendar_event | Cancel a calendar event and notify its attendees. | event_id
list_calendar_events | List your calendar events between two dates. | date_from, date_to
send_email | Send an email now. | to, subject, body
reply_to_email | Reply to a message in your inbox. | message_id, body
search_email | Search your mailbox by sender, subject or words in the message. | query
create_email_draft | Save an email as a draft to finish and send later. | to, subject, body
forward_email | Forward a message from your inbox to someone else. | message_id, to
set_out_of_office_reply | Turn on an automatic away reply for incoming email between two dates. | message, start, end
post_chat_message | Post a message in a team chat channel. | channel, text
send_direct_message | Send a private chat message to one person. | user, text
create_chat_channel | Create a new chat channel and add members. | name, members
search_chat_messages | Search messages in team chat channels and direct messages. | query
set_chat_status | Set your chat status text, optionally until a given time. | text, until
pin_chat_message | Pin a message to the top of a chat channel. | channel, message_id
create_ticket | Open a support ticket about a customer problem or bug. | title, description, priority
assign_ticket | Give a support ticket to a specific agent. | ticket_id, assignee
update_ticket_status | Change a support ticket's status, e.g. open, pending or solved. | ticket_id, status
add_ticket_comment | Add a comment or internal note to a support ticket. | ticket_id, text
search_tickets | Search support tickets by words, customer or status. | query
escalate_ticket | Raise a support ticket to the next support tier with a reason. | ticket_id, reason
search_wiki | Search the internal wiki and handbook pages. | query
create_wiki_page | Write a new page in a wiki space. | title, body, space
update_wiki_page | Change the content of an existing wiki page. | page_id, body
share_document | Give people access to a document with a permission level (view, comment or edit). | doc_id, emails, permission
export_document_pdf | Download a document as a PDF file. | doc_id
comment_on_document | Leave a comment on a document. | doc_id, text
request_time_off | Ask for vacation or other leave on given days. | start, end, type
get_time_off_balance | How many vacation days an employee has left this year. | employee
lookup_employee | Find a colleague's title, team, manager and contact details. | name
start_onboarding | Start the onboarding checklist for a new hire: contract, accounts and laptop. | employee_name, start_date
get_org_chart | Show the reporting lines of a team. | team
submit_performance_review | Submit a performance review with a rating for an employee. | employee, rating, text
create_invoice | Bill a customer: create an invoice with an amount and a due date. | customer, amount_eur, due_date
get_invoice_status | Whether an invoice has been paid, is pending or is overdue. | invoice_id
issue_refund | Return money to a customer for an invoice. | invoice_id, amount_eur, reason
submit_expense_report | Claim back money you spent for work, with a receipt. | amount_eur, category, receipt_url
update_billing_details | Change a customer's billing email, address or VAT number. | customer, billing_email, vat_id
list_overdue_invoices | List the invoices that are past their due date. | customer
run_sql_query | Run a SQL query against the analytics warehouse. | sql
get_dashboard | Open a saved analytics dashboard by name. | name
get_metric | The value of a business metric such as revenue, churn or active users over a period. | metric, date_from, date_to
create_chart | Draw a chart (line, bar or pie) from a data query. | data_query, chart_type
export_report_csv | Download a saved report as a CSV file. | report_id
schedule_report_email | Email a saved report to people on a schedule. | report_id, recipients, frequency
deploy_service | Release a version of a service to an environment. | service, version, environment
rollback_deployment | Go back to a service's previous release in an environment. | service, environment
get_service_logs | Read a service's recent log lines. | service, since
restart_service | Restart a running service in an environment. | service, environment
get_alert_status | List the monitoring alerts that are firing for a service. | service
create_incident | Declare an incident with a severity, page the on-call engineers and open a status page. | title, severity, service
"""
TOOL_TASKS = (
    ("Book the Fjord room tomorrow from 14 to 15 for our design review.", "schedule_meeting_room"),
    ("Put the dentist in my calendar on Friday at 8.", "create_calendar_event"),
    ("Add Pekka and Laura to the Q4 planning meeting I already set up.", "send_meeting_invite"),
    ("When could Anna, Jonas and I all meet for an hour next week?", "find_free_slot"),
    ("Write back to Tom's last email and tell him Thursday works.", "reply_to_email"),
    ("Prepare an email to the board about the budget, but don't send it yet.", "create_email_draft"),
    ("I'm on holiday from 1 to 12 December; let people who write to me know.", "set_out_of_office_reply"),
    ("Tell the sales channel that the new price list is live.", "post_chat_message"),
    ("Let Mikko know privately that the build is green.", "send_direct_message"),
    ("Did anyone on chat mention the Lisbon offsite?", "search_chat_messages"),
    ("The payment page throws a 500 error for some customers; log it for the support team.", "create_ticket"),
    ("Hand ticket 4812 over to Sofia.", "assign_ticket"),
    ("Mark ticket 4812 as solved.", "update_ticket_status"),
    ("This customer has waited a week and is furious; get the next level of support on ticket 4812.", "escalate_ticket"),
    ("What does our handbook say about parental leave?", "search_wiki"),
    ("Give the auditors read access to the Q3 controls spreadsheet.", "share_document"),
    ("I need the onboarding guide as a PDF to send to a contractor.", "export_document_pdf"),
    ("How many vacation days do I have left?", "get_time_off_balance"),
    ("I'm taking next Monday off.", "request_time_off"),
    ("Who is Sara Lindqvist's manager?", "lookup_employee"),
    ("Elias starts with us on 3 November; get his paperwork and laptop going.", "start_onboarding"),
    ("Bill Kestrel Oy 12,400 euros for the October consulting work, due in 30 days.", "create_invoice"),
    ("Has Brightdesk paid INV-2291 yet?", "get_invoice_status"),
    ("Give the customer their money back for the double charge on INV-2291.", "issue_refund"),
    ("Here's my taxi receipt from the client visit, 38 euros.", "submit_expense_report"),
    ("Which customers owe us money past the due date?", "list_overdue_invoices"),
    ("What was our churn in September?", "get_metric"),
    ("Send the weekly sales report to the leadership team every Monday.", "schedule_report_email"),
    ("Ship version 2.14 of the billing service to production.", "deploy_service"),
    ("The last checkout release broke production; go back to the one before.", "rollback_deployment"),
    ("Show me what the search API printed in the last 15 minutes.", "get_service_logs"),
    ("Is anything going off for the payments service right now?", "get_alert_status"),
    ("The whole site is down, open a sev-1.", "create_incident"),
    ("Move the Nordvik deal to negotiation.", "crm_update_deal_stage"),
    ("I just spoke with the CFO of Aurora Labs on the phone; note what we discussed.", "crm_log_call"),
    ("What's the phone number of our contact at Cedar Systems?", "crm_search_contacts"),
    ("Give me the full picture of the Boreal Group account: people, deals and what happened lately.", "crm_get_account"),
    ("Set my status to 'in workshops' until 5 pm.", "set_chat_status"),
    ("Write up the incident postmortem as a new page in the Ops space.", "create_wiki_page"),
    ("Plot weekly signups for the last quarter as a line.", "create_chart"),
)


def tool_catalog() -> list[dict]:
    """About 60 tools of a workplace assistant as OpenAI function schemas; every parameter is a string."""
    out = []
    for line in _CATALOG.strip().splitlines():
        name, description, params = (p.strip() for p in line.split("|"))
        props = {p: dict(type="string") for p in params.split(", ")}
        out.append(dict(type="function", function=dict(name=name, description=description,
                                                       parameters=dict(type="object", properties=props, required=list(props)))))
    return out


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "connect":
        print(f"connected as {connect(sys.argv[2], sys.argv[3])}")
    elif sys.argv[1:] == ["fetch"]:
        print(f"data in {fetch()}")
        from huggingface_hub import snapshot_download
        print(f"question encoder in {snapshot_download(QUERY_MODEL, revision=QUERY_REVISION)}")
    else:
        sys.exit(__doc__)
