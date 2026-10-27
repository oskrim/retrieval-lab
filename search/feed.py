"""What Vespa holds: documents(ds, fields) yields every Vespa document of dataset ds as (document id, its fields), one
value for each document field app/schema.sd declares (fields: name -> type, as the schema writes it). lab vespa up
feeds them, and feeds again after this file, a module of this folder it imports, or the schema's document fields
change; a data file it reads is not tracked (lab vespa down, then lab vespa up, feeds from scratch).

As it is, one post is one document: its title and body as they are, and the vectors that come with the data (lab
setup downloads them; nothing is embedded here, so Vespa needs no GPU). The id field holds the post's id: search.py
turns hits into rows of ds by it. Things to try:
- What title and body hold: the body cleaned up, the title repeated, words from analyze.py's synonyms added.
- Chunks: several documents per post, each with the post's id in its id field (see chunks() below); search.py keeps
  each post's best chunk (HITS_PER_POST there).
- New vectors: lab.embed_documents(texts) embeds texts the way the data's document vectors were (pplx-embed-v1-0.6b,
  mean pooling, up to 8,192 tokens; texts as "<title>\\n<body>"). It needs a GPU: FiQA's 57,912 posts take minutes on
  one and hours on a CPU. Grading runs lab vespa up within 30 minutes, on a box without a GPU unless the organizers
  say otherwise, so ask them before handing in a feed.py that embeds.
"""

from collections.abc import Iterator

import numpy as np

import lab
import labvespa

CELLS = {"dense_bits": "int8", "late_bits": "int8", "late_int8": "int8"}  # field -> cells it needs


def documents(ds: lab.Dataset, fields: dict[str, str]) -> Iterator[tuple[str, dict]]:
    value = values(ds, fields)
    for i, post in enumerate(ds.ids):
        yield post, value(i)


def values(ds: lab.Dataset, fields: dict[str, str]):
    """value(i): the fields of post i that the schema declares and the data has values for."""
    for f, cells in CELLS.items():
        if f in fields and labvespa.cell_type(fields[f]) != cells:
            raise ValueError(f"schema.sd field {f} needs {cells} cells: feed.py writes {cells} values into it")
    floats = lab.vectors(ds)[0] if "dense_float" in fields else None
    bits = lab.dense_parts(ds)[1].view(np.uint8).view(np.int8) if "dense_bits" in fields else None
    tokens = lab.late_tokens(ds) if {"late_bits", "late_int8", "late_scale"} & set(fields) else None
    late = labvespa.late_bits(ds)[0] if "late_bits" in fields else None

    def blocks(x: np.ndarray, i: int) -> dict:  # one hex block per token of post i
        return {"blocks": {str(t): x[a].tobytes().hex() for t, a in enumerate(range(tokens.start[i], tokens.start[i + 1]))}}

    writers = {  # field -> its value for post i
        "id": lambda i: ds.ids[i],
        "title": lambda i: ds.titles[i],
        "body": lambda i: ds.bodies[i],
        "dense_float": lambda i: {"values": labvespa.hexed(floats[i], labvespa.cell_type(fields["dense_float"]))},
        "dense_bits": lambda i: {"values": bits[i].tobytes().hex()},
        "late_bits": lambda i: blocks(late, i),
        "late_int8": lambda i: blocks(tokens.docs, i),
        "late_scale": lambda i: {"cells": {str(t): float(tokens.scale[a])
                                           for t, a in enumerate(range(tokens.start[i], tokens.start[i + 1]))}},
    }
    feeding = [(f, w) for f, w in writers.items() if f in fields]
    return lambda i: {f: w(i) for f, w in feeding}


# Chunks: to feed each post as pieces of about CHUNK_WORDS words of its body, make documents() yield from chunks()
# instead. Each chunk keeps the post's id in its id field and the post's vectors; vectors of the chunk itself would
# come from lab.embed_documents. In search.py, set HITS_PER_POST to about the number of chunks a post has.
CHUNK_WORDS = 200


def chunks(ds: lab.Dataset, fields: dict[str, str]) -> Iterator[tuple[str, dict]]:
    value = values(ds, fields)
    for i, post in enumerate(ds.ids):
        words = ds.bodies[i].split()
        for c, a in enumerate(range(0, max(len(words), 1), CHUNK_WORDS)):
            yield f"{post}#{c}", {**value(i), "body": " ".join(words[a:a + CHUNK_WORDS])}
