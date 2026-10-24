"""Your search: find candidates fast, score them carefully, return the best k.

  lab eval search.py                       nDCG@10 on the dev questions, with a 95% interval
  lab eval search.py --worst 5             ... and the 5 worst questions with your top 10
  lab eval search.py --data caiq           only one dataset (fiqa or caiq)
  lab submit search.py -m "what changed"   runs it on the leaderboard's FiQA questions

As it is, candidates are the notebook's hybrid (BM25 + dense vectors, fused), and the score mixes the hybrid's order
with the cross-encoder's, as the notebook does: alone, the cross-encoder ranks below the hybrid it reranks.
Things to try: other weights or n in candidates; the cross-encoder's weight in score; a different reranker
(lab.rerank_local's model); scoring fewer or more candidates. search must return k distinct
document rows (indices into ds.ids, ds.titles and ds.bodies), best first.
"""

import numpy as np

import lab


def candidates(ds: lab.Dataset, question: str) -> np.ndarray:
    """Up to 100 likely documents, best first."""
    return lab.hybrid(ds, question, bm25_weight=1.0, dense_weight=4.0, n=100)


def minmax(x: np.ndarray) -> np.ndarray:
    """x scaled to 0..1, so scores on different scales can be added."""
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() or 1.0)


def score(ds: lab.Dataset, question: str, rows: np.ndarray) -> np.ndarray:
    """One score per row; higher is better. The cross-encoder reads the question and each document together; the
    hybrid's order (rows come best first) keeps what it knew."""
    return minmax(-np.arange(len(rows))) + 4 * minmax(lab.rerank_local(ds, question, rows))


def search(ds: lab.Dataset, question: str, k: int = 10) -> np.ndarray:
    rows = candidates(ds, question)
    return rows[lab.argsort_desc(score(ds, question, rows))][:k]
