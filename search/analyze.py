"""How search.py reads a question: its words, which of them are stopwords, their stems, and the synonym phrases in it
(the groups in synonyms.json beside this file). feed.py may use it for the documents too.

As it is, a question is read as the notebook's lab.hybrid reads it: lowercase words without accents, the stopwords
below dropped, Snowball (Porter 2) stems, synonym phrases found longest first over stems. Vespa analyzes the
documents' title and body itself (its own tokens and stems); search.py sends it the question's words as they are.
Things to try:
- STOPWORDS: fewer or more. Vespa counts every word of a document, so a dropped word only changes the question.
- stem: another snowballstemmer language or algorithm, or none (return the word itself).
- synonyms.json: groups of phrases that mean the same; search.py sends each phrase it finds as one equiv term.
"""

import hashlib
import json
import math
import pathlib
import re
import threading
import unicodedata
from collections import Counter

import numpy as np
import snowballstemmer

import lab

WORD = re.compile(r"[^\W_]+")  # runs of Unicode letters and digits; everything else separates words
STOPWORDS = frozenset("""a an and are as at be been but by can could did do does for from had has have how i if in into is
it its me my of on or our over should so than that the their them then there these they this those to was we were what
when where which while who why will with would you your""".split())
SYNONYMS = pathlib.Path(__file__).with_name("synonyms.json")
_STEMMER = snowballstemmer.stemmer("english")
_STEMS: dict[str, str] = {}  # every word is stemmed once
_LOCK = threading.Lock()  # the stemmer keeps its state between calls, and the searches run on threads


def normalize(text: str) -> str:
    """Lowercase, accents removed (é -> e), compatibility forms folded (ﬁ -> fi)."""
    if text.isascii():
        return text.lower()
    nfkd = unicodedata.normalize("NFKD", text)
    return nfkd.lower() if nfkd.isascii() else "".join(c for c in nfkd if not unicodedata.combining(c)).casefold()


def tokenize(text: str) -> list[str]:
    return WORD.findall(normalize(text))


def stem(word: str) -> str:
    """The word's Snowball (Porter 2) stem: encrypt, encrypted, encryption -> encrypt."""
    if word not in _STEMS:
        with _LOCK:
            _STEMS[word] = _STEMMER.stemWord(word)
    return _STEMS[word]


def term(token: str) -> str | None:
    """The term a token counts as: its stem, or None for a stopword."""
    return None if token in STOPWORDS else stem(token)


# The synonym groups as token lists, and every alternative's stems -> its group (the first group wins a duplicate).
GROUPS = [[tokenize(p) for p in group["phrases"] if tokenize(p)] for group in json.loads(SYNONYMS.read_text())["groups"]]
PHRASES: dict[tuple[str, ...], int] = {}
for _g, _alternatives in enumerate(GROUPS):
    for _tokens in _alternatives:
        PHRASES.setdefault(tuple(stem(t) for t in _tokens), _g)
LONGEST = max(map(len, PHRASES), default=0)


def concepts(tokens: list[str]) -> list[tuple[int, int, int]]:
    """(group, position, length) of every synonym phrase in tokens: longest match first, left to right, no overlaps."""
    stems, found, i = [stem(t) for t in tokens], [], 0
    while i < len(stems):
        for n in range(min(LONGEST, len(stems) - i), 0, -1):
            g = PHRASES.get(tuple(stems[i:i + n]))
            if g is not None:
                found.append((g, i, n))
                i += n
                break
        else:
            i += 1
    return found


def idf(ds: lab.Dataset, t: str) -> float:
    """The idf of term t (a stem) over the documents' title and body, as BM25 weighs it: log(1 + (n - df + 0.5) /
    (df + 0.5)) for df of the n documents containing it."""
    df = document_frequency(ds)
    return math.log(1 + (len(ds.ids) - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))


_DF: dict[tuple[str, int], Counter] = {}
_DF_LOCK = threading.Lock()


def document_frequency(ds: lab.Dataset) -> Counter:
    """term -> the number of ds's documents whose title or body contains it; built once per dataset and kept in
    data/cache until this file changes."""
    with _DF_LOCK:
        if (ds.name, len(ds.ids)) not in _DF:
            key = f"{lab.DATA_VERSION} {ds.name} {len(ds.ids)} {pathlib.Path(__file__).read_text()}"
            small = sum(map(len, ds.title_body)) < lab.PERSIST_CHARS  # builds in milliseconds; not kept
            name = None if small else f"analyze-df-{hashlib.sha256(key.encode()).hexdigest()[:24]}"

            def build() -> dict[str, np.ndarray]:
                counts = Counter()
                for text in ds.title_body:
                    counts.update({w for w in map(term, tokenize(text)) if w is not None})
                return dict(terms=np.array(list(counts)), df=np.array(list(counts.values()), dtype=np.int64))

            a = lab.disk_cached(name, build)
            _DF[ds.name, len(ds.ids)] = Counter(dict(zip(a["terms"].tolist(), a["df"].tolist())))
        return _DF[ds.name, len(ds.ids)]
