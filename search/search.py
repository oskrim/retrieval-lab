"""Your search on your own Vespa: the query, sent to the rank profiles in app/schema.sd. Everything in this folder
(search/) is graded together: search.py, feed.py (what Vespa holds), analyze.py (how a question is read), synonyms.json
and app/.

  lab eval search                          nDCG@10 on the dev questions, saved as a run; it deploys app/ first when
                                           app/ changed (about 4 s)
  lab vespa up                             feed again after feed.py or the schema's document fields change (about 2
                                           minutes; lab eval search says when)
  lab compare                              what changed since the run before (lab compare question N: one question)
  lab submit search                        send search/ for grading

As it is, the query finds the documents with any of the question's words or synonym phrases (synonyms.json's groups as
equiv terms) or among the nearest by vector, sends the question's word pairs as phrases and its vector, and Vespa ranks
them with notebook 15's hybrid (late interaction waits commented out). app/schema.sd's common profile has the
notebook's pieces as functions (saturated BM25, the phrase share, cosine, the bits, late interaction); a profile of
your own combines them. Things to try:
- PROFILE: your profile; INPUTS: its query(...) inputs, per query, without a deploy. query(saturation), for one:
  Vespa counts stopwords in a field's length, so its BM25 scores of long documents run higher than the notebook's.
- Phases: a first phase that is cheap over every match, a second-phase (rerank-count) for the expensive part, a
  global-phase across the best hits.
- Matching: every document with a question word is ranked. ({targetHits:100}weakAnd(...)) instead of or ranks
  fewer, faster, but it picks them by its own term weights, not BM25.
- Binary against float: nearestNeighbor(dense_bits, qbits) finds candidates by hamming distance; dense_float holds the
  float vectors to rescore them with, and dense (float question × bits) rescores without float vectors in memory.
- Memory: attribute: paged on a vector field, bfloat16 cells, or HNSW (index: hnsw) against exact search;
  lab vespa status shows the memory.
- HITS_PER_POST: with chunks (feed.py), Vespa returns a post once per matching chunk; asking for this many hits per
  row keeps k distinct posts after the best chunk of each is kept.
search must return k distinct document rows (indices into ds.ids), best first.
"""

import json

import numpy as np

import analyze
import lab
import labvespa

PROFILE = "hybrid"
INPUTS = {}
LATE = False  # send the question's late-interaction tokens (query(qlate_float), query(qlate_bits)) that late uses
TARGET_HITS = 100  # how many documents nearestNeighbor finds
WORDS_ONLY: set[str] = set()  # profiles that match only the question's words; dense-* match only by vector
NEAREST: dict[str, tuple[str, str]] = {}  # profile -> (field, input) its nearestNeighbor uses; others: the bits
HITS_PER_POST = 1


def words(question: str) -> tuple[list[str], list[list[str]]]:
    """The question's words without stopwords and outside its synonym phrases, each stem once (as lab.hybrid's BM25
    counts them), and its word pairs: neighbouring words with the stopwords between them, which a document must
    contain as a phrase."""
    tokens = analyze.tokenize(question)
    inside = {i + k for _, i, n in analyze.concepts(tokens) for k in range(n)}
    kept = [i for i, t in enumerate(tokens) if analyze.term(t) is not None]
    stems = {}
    for i in kept:
        if i not in inside:
            stems.setdefault(analyze.term(tokens[i]), tokens[i])
    return list(stems.values()), [tokens[a:b + 1] for a, b in zip(kept, kept[1:])]


def synonyms(ds: lab.Dataset, question: str) -> list[str]:
    """Each synonym phrase in the question (synonyms.json's groups, found by analyze.concepts) as one equiv term of its
    group's phrases: BM25 counts a match of any of them as the same term. Its significance, which Vespa's bm25 uses as
    the term's idf, is the idf of the question's own wording, as lab's BM25 weighs a synonym."""
    tokens, out = analyze.tokenize(question), []
    for g, i, n in analyze.concepts(tokens):
        own = {w for w in map(analyze.term, tokens[i:i + n]) if w is not None}
        alternatives = [json.dumps(p[0]) if len(p) == 1 else f"phrase({', '.join(map(json.dumps, p))})"
                        for p in analyze.GROUPS[g]]
        term = f"equiv({', '.join(alternatives)})" if len(alternatives) > 1 else alternatives[0]
        out.append(f"default contains ({{significance: {sum(analyze.idf(ds, t) for t in own):.4f}}}{term})")
    return out


def yql(ds: lab.Dataset, question: str, terms: list[str], pairs: list[list[str]]) -> str:
    """The query: the documents with any of the question's words or synonym phrases, or among the TARGET_HITS nearest
    by vector; the word pairs as phrases that only rank (rank() matches by its first argument); never the question's
    hidden documents."""
    lexical = not PROFILE.startswith("dense-")
    items = [f"default contains {json.dumps(t)}" for t in terms] + synonyms(ds, question)
    match = [f"({' or '.join(items)})"] if lexical and items else []
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
              "query(pair_count)": len(pairs), **INPUTS}
    if LATE:
        inputs.update(labvespa.late_inputs(ds, question))
    hits = labvespa.query(ds.name, yql(ds, question, terms, pairs), PROFILE, inputs, hits=k * HITS_PER_POST)
    return np.array([ds.row[d] for d, _ in best_chunks(hits)], dtype=np.int64)[:k]


def best_chunks(hits: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Each post once, at its best hit: with chunks (feed.py) several hits can name the same post."""
    best: dict[str, float] = {}
    for post, score in hits:
        best.setdefault(post, score)
    return list(best.items())
