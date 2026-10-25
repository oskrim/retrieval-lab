"""Your search on your own Vespa: the notebook's ranking as rank profiles in vespa/app/schemas/fiqa.sd.

  lab vespa up                             start Vespa, deploy vespa/app and feed the data (the feed runs once)
  lab vespa deploy                         after editing vespa/app/schemas/fiqa.sd (caiq.sd is made from it)
  lab eval vespa/search.py                 nDCG@10 on the dev questions
  lab submit vespa/search.py -m "what changed"

As it is, Vespa ranks with the hybrid profile, the notebook's search before reranking (lab.hybrid): saturated BM25 of
title and body, the share of the question's word pairs found as phrases, and the question's vector against the
documents' sign bits. Things to try:
- PROFILE: bm25-body, bm25-fields, phrase, dense-float, dense-int8, dense-binary, hybrid, maxsim, maxsim-int8, ltr
  (a LightGBM model in the global phase; tools/vespa_ltr.py trains it).
- INPUTS: the profiles' query(...) inputs, per query, without a deploy. query(saturation), for one: Vespa counts
  stopwords in a field's length, so its BM25 scores of long documents run higher than the notebook's.
- Phases: a first phase that is cheap over every match, a second-phase (rerank-count) for the expensive part, a
  global-phase across the best hits (maxsim's late interaction does this).
- Binary against float: dense-binary finds candidates by hamming distance and rescores them with the float vectors;
  rescoring with dense (float question × bits) instead needs no float vectors in memory at all.
- Memory: attribute: paged on a vector field, bfloat16 cells, or HNSW (index: hnsw) against exact search;
  lab vespa status shows the memory.
- RERANK: reorder Vespa's best documents with the cross-encoder (lab.rerank_local), as templates/search.py does.
search must return k distinct document rows (indices into ds.ids), best first.
"""

import json

import numpy as np

import lab
import labvespa

PROFILE = "hybrid"
INPUTS = {"query(bm25_weight)": 1.0, "query(dense_weight)": 4.0, "query(phrase_weight)": 0.2, "query(late_weight)": 0.5}
TARGET_HITS = 100  # how many documents nearestNeighbor finds
RERANK = 0  # above 0: the cross-encoder reorders Vespa's best RERANK documents
WORDS_ONLY = {"bm25-body", "bm25-fields", "phrase"}  # profiles that match only the question's words; dense-* only vectors
NEAREST = {"dense-float": ("dense_float", "q"), "dense-int8": ("dense_int8", "q")}  # (field, input); others: the bits


def words(question: str) -> tuple[list[str], list[list[str]]]:
    """The question's words without stopwords, each stem once (as lab.hybrid's BM25 counts them), and its word pairs:
    neighbouring words with the stopwords between them, which a document must contain as a phrase."""
    tokens = lab.tokenize(question)
    kept = [i for i, t in enumerate(tokens) if t not in lab.STOPWORDS]
    stems = {}
    for i in kept:
        stems.setdefault(lab.stem(tokens[i]), tokens[i])
    return list(stems.values()), [tokens[a:b + 1] for a, b in zip(kept, kept[1:])]


def yql(ds: lab.Dataset, question: str, terms: list[str], pairs: list[list[str]]) -> str:
    """The query: the documents with any of the question's words, or among the nearest by vector; the word pairs as
    phrases that only rank (rank() matches by its first argument); never the question's hidden documents."""
    lexical = not PROFILE.startswith("dense-")
    match = ["userInput(@q)"] if lexical and terms else []
    if PROFILE not in WORDS_ONLY:
        field, vector = NEAREST.get(PROFILE, ("dense_bits", "qbits"))
        match.append(f"({{targetHits:{TARGET_HITS}}}nearestNeighbor({field}, {vector}))")
    where = " or ".join(match) or "false"
    hidden = ds.hidden_by_text.get(question)
    if hidden:
        where = f"({where}) and !(id in ({', '.join(json.dumps(d) for d in sorted(hidden))}))"
    phrases = [f"{field} contains phrase({', '.join(map(json.dumps, p))})" for p in pairs for field in ("title_pairs", "body_pairs")]
    if phrases and lexical:
        where = f"rank({where}, {', '.join(phrases)})"
    return f"select * from {ds.name} where {where}"


def search(ds: lab.Dataset, question: str, k: int = 10) -> np.ndarray:
    terms, pairs = words(question)
    mean, _, known = lab.dense_parts(ds)
    q = known[question] if question in known else lab.embed([question])[0]  # the question's unit pplx vector
    inputs = {"query(q)": q, "query(qc)": q - mean, "query(qbits)": lab.pack_signs(q - mean >= 0).view(np.int8),
              "query(pairs)": len(pairs), **INPUTS}
    if PROFILE.startswith("maxsim"):
        inputs.update(labvespa.late_inputs(ds, question))
    hits = labvespa.query(ds.name, yql(ds, question, terms, pairs), PROFILE, inputs, hits=max(k, RERANK),
                          q=" ".join(terms))
    rows = np.array([ds.row[d] for d, _ in hits], dtype=np.int64)
    if RERANK:
        top = rows[:RERANK]
        mixed = minmax(-np.arange(len(top))) + 4 * minmax(lab.rerank_local(ds, question, top))
        rows = np.concatenate([top[lab.argsort_desc(mixed)], rows[RERANK:]])
    return rows[:k]


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() or 1.0)
