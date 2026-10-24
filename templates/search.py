"""Your search: find candidates fast, score them carefully, return the best k.

  lab eval search.py                       nDCG@10 on the dev questions, with a 95% interval
  lab eval search.py --worst 5             ... and the 5 worst questions with your top 10
  lab submit search.py -m "what changed"   runs it on the leaderboard's FiQA questions

As it is, candidates are the notebook's hybrid (BM25 + dense vectors, fused) and the score is the cross-encoder alone.
Things to try: other weights or n in candidates; mixing the hybrid's order into score (the cross-encoder does not see
it); a different reranker (lab.rerank_local's model); scoring fewer or more candidates. search must return k distinct
document rows (indices into ds.ids, ds.titles and ds.bodies), best first.
"""

import numpy as np

import lab


def candidates(ds: lab.Dataset, question: str) -> np.ndarray:
    """Up to 100 likely documents, best first."""
    return lab.hybrid(ds, question, bm25_weight=1.0, dense_weight=4.0, n=100)


def score(ds: lab.Dataset, question: str, rows: np.ndarray) -> np.ndarray:
    """One score per row; higher is better. The cross-encoder reads the question and each document together."""
    return lab.rerank_local(ds, question, rows)


def search(ds: lab.Dataset, question: str, k: int = 10) -> np.ndarray:
    rows = candidates(ds, question)
    return rows[lab.argsort_desc(score(ds, question, rows))][:k]
