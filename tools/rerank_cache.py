"""Maintainer script: precomputes the reranker's scores for every question the lab asks, so machines without a GPU
can rerank from the cache (lab.rerank_local). Run it on a GPU machine with the data downloaded (lab setup):

  python tools/rerank_cache.py --board-questions board.json [--model Qwen/Qwen3-Reranker-0.6B]

Questions: fiqa dev and test, caiq test, and the leaderboard's (board.json: [{"id", "text"}], fetched from the server).
Documents per question: the top 100 of lab.hybrid as the template calls it, with BM25 only and with dense only, the
notebook's top 20 (lab.hybrid with n=20), and the rows in data/intent-top20.npz. Writes
data/<data version>/rerank-<model>.npz, the release asset lab setup downloads: questions (sha256 of the text, so the
board's questions do not ship), offsets, docs, scores (log-odds, float32). Scores go to data/rerank-work-<model>.sqlite
as they are made, so a stopped run resumes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys
import time

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import lab  # noqa: E402

console = Console(highlight=False)
CHUNK = 8192  # pairs tokenized and committed at a time


def questions(board: pathlib.Path, limit: int | None) -> list[tuple[lab.Dataset, str, str | None]]:
    """(dataset, question text, test question id or None) for every question the lab asks."""
    out = []
    for name, split in (("fiqa", "dev"), ("fiqa", "test"), ("caiq", "test")):
        ds = lab.load(name, split)
        out += [(ds, ds.queries[q], q if split == "test" else None) for q in ds.qids[:limit]]
    rows = json.loads(board.read_text())
    rows = rows["questions"] if isinstance(rows, dict) else rows
    fiqa = next(ds for ds, _, _ in out if ds.name == "fiqa")
    return out + [(fiqa, r["text"], None) for r in rows[:limit]]


def pools(asked: list[tuple[lab.Dataset, str, str | None]], top20: pathlib.Path) -> dict[str, tuple[str, set[str]]]:
    """question key -> (question text, the doc ids to score)."""
    extra = dict(np.load(top20)) if top20.exists() else {}
    if not extra:
        console.print(f"[yellow]![/] no {top20}; its rows are left out")
    out: dict[str, tuple[str, set[str]]] = {}
    with progress() as bar:
        task = bar.add_task("candidates", total=len(asked))
        for ds, text, qid in asked:
            rows = [lab.hybrid(ds, text, n=100), lab.hybrid(ds, text, 1.0, 0.0, n=100), lab.hybrid(ds, text, 0.0, 1.0, n=100),
                    lab.hybrid(ds, text, n=20)]
            if qid is not None and ds.name in extra:
                rows.append(extra[ds.name][ds.qrow[qid]])
            key = lab.question_key(text)
            out.setdefault(key, (text, set()))[1].update(ds.ids[i] for r in rows for i in r)
            bar.advance(task)
    return out


def progress() -> Progress:
    return Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                    TimeRemainingColumn(), console=console)


def score(work: sqlite3.Connection, todo: list[tuple[str, str, str, str]], model: str, device: str, batch_tokens: int) -> None:
    """Scores (key, doc id, question, doc text) pairs into work: CHUNK at a time, in batches of similar length."""
    todo = sorted(todo, key=lambda p: len(p[3]))
    start, done = time.time(), 0
    with progress() as bar:
        task = bar.add_task("scoring", total=len(todo))
        for a in range(0, len(todo), CHUNK):
            chunk = todo[a:a + CHUNK]
            inputs = lab.reranker_inputs(model, device, [(q, d) for _, _, q, d in chunk])
            rows = []
            for batch in batches([len(x) for x in inputs], batch_tokens):
                for i, s in zip(batch, lab.reranker_scores(model, device, [inputs[i] for i in batch])):
                    rows.append((chunk[i][0], chunk[i][1], s))
                bar.advance(task, len(batch))
                done += len(batch)
                bar.update(task, description=f"scoring · {done / (time.time() - start):,.0f} pairs/s")
            with work:
                work.executemany("INSERT OR REPLACE INTO scores VALUES (?, ?, ?)", rows)
    if done:
        console.print(f"{done:,} pairs in {time.time() - start:,.0f} s: {done / (time.time() - start):,.0f} pairs/s")


def batches(lengths: list[int], budget: int) -> list[list[int]]:
    """Indices into lengths, shortest first, in batches whose size times their longest length fits budget."""
    out, batch = [], []
    for i in sorted(range(len(lengths)), key=lengths.__getitem__):
        if batch and (len(batch) + 1) * lengths[i] > budget:
            out.append(batch)
            batch = []
        batch.append(i)
    return out + [batch] if batch else out


def export(work: sqlite3.Connection, keys: list[str], path: pathlib.Path) -> None:
    keys = sorted(keys)
    offsets, docs, scores = [0], [], []
    for k in keys:
        rows = work.execute("SELECT doc, score FROM scores WHERE question = ? ORDER BY doc", (k,)).fetchall()
        docs += [d for d, _ in rows]
        scores += [s for _, s in rows]
        offsets.append(len(docs))
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, questions=np.array(keys), offsets=np.array(offsets, dtype=np.int64), docs=np.array(docs),
                        scores=np.array(scores, dtype=np.float32))
    tmp.rename(path)
    console.print(f"[green]✓[/] {path} · {len(keys):,} questions · {len(docs):,} pairs · {path.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--board-questions", type=pathlib.Path, required=True, help='JSON: [{"id", "text"}]')
    p.add_argument("--model", default=lab.RERANK_MODEL)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-tokens", type=int, default=32768, help="tokens per GPU batch (default 32768)")
    p.add_argument("--top20", type=pathlib.Path, default=lab.DATA / "intent-top20.npz")
    p.add_argument("--limit", type=int, default=None, help="only the first N questions of each source (a test run)")
    p.add_argument("--out", type=pathlib.Path, default=None, help="default: data/<data version>/rerank-<model>.npz")
    args = p.parse_args()

    out = args.out or lab.fetch() / lab.rerank_cache_name(args.model)
    work = sqlite3.connect(lab.DATA / f"rerank-work-{args.model.split('/')[-1]}.sqlite")
    work.execute("CREATE TABLE IF NOT EXISTS scores (question TEXT, doc TEXT, score REAL, PRIMARY KEY (question, doc))")
    pool = pools(questions(args.board_questions, args.limit), args.top20)
    texts = {}
    for name in ("fiqa", "caiq"):
        ds = lab.load(name)
        texts.update((d, lab.rerank_text(ds, i)) for i, d in enumerate(ds.ids))
    todo = []
    for key, (question, docs) in pool.items():
        have = {d for (d,) in work.execute("SELECT doc FROM scores WHERE question = ?", (key,))}
        todo += [(key, d, question, texts[d]) for d in sorted(docs - have)]
    total = sum(len(docs) for _, docs in pool.values())
    console.print(f"{len(pool):,} questions · {total:,} pairs · {total - len(todo):,} already scored · model {args.model}")
    score(work, todo, args.model, args.device, args.batch_tokens)
    export(work, list(pool), out)


if __name__ == "__main__":
    main()
