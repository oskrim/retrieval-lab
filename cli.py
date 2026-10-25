"""The lab CLI: set up the GPU box, evaluate and submit your search.py or agent.py, watch the leaderboards."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (BarColumn, DownloadColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TransferSpeedColumn)
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

import lab
import labvespa

HOME = pathlib.Path(os.environ.get("LAB_HOME", "~/.lab")).expanduser()
VENV = HOME / "vllm-venv"
SERVE_LOG, SERVE_PID, EVALS = HOME / "serve.log", HOME / "serve.pid", HOME / "evals.jsonl"
TEMPLATES = lab.HERE / "templates"
MIN_FREE_GB = 60
LABEL_CHARS = 60  # the lab's limit on a submission's label
EVAL_SETS = (("fiqa", "dev"), ("caiq", "test"))  # caiq has no dev split
EVAL_LABELS = [f"{name} {split}" for name, split in EVAL_SETS]
BOARD_DATASET = "fiqa"
console = Console(theme=Theme({"ok": "green", "err": "bold red", "warn": "yellow", "acc": "cyan", "dim": "grey50",
                               "num": "bold", "me": "bold cyan", "grade2": "bold green"}), highlight=False)


class Fail(Exception):
    """A short message for the participant; the CLI exits with 1."""


class UserCodeError(Exception):
    """The participant's file raised exc."""

    def __init__(self, exc: BaseException, path: pathlib.Path) -> None:
        super().__init__(str(exc))
        self.exc, self.path = exc, path


class ContractError(Exception):
    pass


def ok(msg: str) -> None:
    console.print(f"[ok]✓[/] {msg}")


def fail(msg: str) -> None:
    console.print(Text.assemble(("✗ ", "err"), msg))


def gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


# ---- participant files

def load_file(name: str) -> tuple[pathlib.Path, object]:
    """Imports the participant's file by path; its directory goes on sys.path so its own helpers import too."""
    path = pathlib.Path(name).resolve()
    if not path.is_file() or path.suffix != ".py":
        raise Fail(f"no Python file {name}")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"lab_user_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    with console.status(f"loading {path.name}"):
        try:
            spec.loader.exec_module(module)
        except (Exception, SystemExit) as e:
            raise UserCodeError(e, path) from None
    return path, module


def file_kind(module) -> str:
    """"search" for a file with search(), "agent" for one with answer()."""
    has = {k for k in ("search", "answer") if callable(getattr(module, k, None))}
    if has == {"search"}:
        return "search"
    if has == {"answer"}:
        return "agent"
    if has:
        raise Fail("the file defines both search() and answer(); keep one per file")
    raise Fail("the file defines neither search(ds, question, k) nor answer(question)")


LAB_FILES = {pathlib.Path(lab.__file__).resolve(), pathlib.Path(__file__).resolve()}


def theirs(filename: str, path: pathlib.Path) -> bool:
    """Whether filename is the participant's code: their file, or a module of theirs beside it (not lab's own)."""
    if filename.startswith("<"):
        return False
    p = pathlib.Path(filename).resolve()
    return p == path or (p.is_relative_to(path.parent) and p not in LAB_FILES and "site-packages" not in p.parts)


def user_frames(exc: BaseException, path: pathlib.Path) -> list[traceback.FrameSummary]:
    """The traceback frames of exc that are in the participant's code, and the line of a syntax error there."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if theirs(f.filename, path)]
    if isinstance(exc, SyntaxError) and exc.filename and theirs(exc.filename, path):
        frames.append(traceback.FrameSummary(exc.filename, exc.lineno or 0, "<module>", line=(exc.text or "").strip()))
    return frames


def show_user_error(err: UserCodeError) -> None:
    exc = err.exc
    console.print(f"[err]✗ {err.path.name} raised {type(exc).__name__}[/]")
    for f in user_frames(exc, err.path):
        console.print(f"  [acc]{pathlib.Path(f.filename).name}:{f.lineno}[/] in {f.name}")
        if f.line:
            console.print(Syntax(f.line, "python", theme="ansi_dark", background_color="default", padding=(0, 0, 0, 4)))
    console.print(f"  {type(exc).__name__}: {exc}", markup=False)


def check_rows(rows, n_docs: int, k: int) -> list[int]:
    """search's contract: k distinct document rows (ints in range), best first."""
    if isinstance(rows, (set, frozenset)):
        raise ContractError(f"search returned a set, which has no order; return a list or array of {k} rows, best first")
    try:
        items = list(rows)
    except TypeError:
        raise ContractError(f"search returned {type(rows).__name__}; return a list or array of {k} document rows") from None
    bad = [x for x in items if isinstance(x, bool) or not isinstance(x, (int, np.integer))]
    if bad:
        raise ContractError(f"search returned {type(bad[0]).__name__} {bad[0]!r}; rows are ints (indices into ds.ids)")
    if len(items) != k:
        raise ContractError(f"search returned {len(items)} rows; return exactly k = {k}")
    if len(set(map(int, items))) != len(items):
        raise ContractError("search returned the same row twice")
    out = [int(x) for x in items]
    if not all(0 <= x < n_docs for x in out):
        raise ContractError(f"row {next(x for x in out if not 0 <= x < n_docs)} is out of range; rows go from 0 to {n_docs - 1}")
    return out


def check_deals(found) -> set[str]:
    """answer's contract: a set of deal ids (str)."""
    if not isinstance(found, (set, frozenset)):
        raise ContractError(f"answer returned {type(found).__name__}; return a set of deal ids such as {{'D0042'}}")
    bad = [x for x in found if not isinstance(x, str)]
    if bad:
        raise ContractError(f"answer returned {type(bad[0]).__name__} {bad[0]!r} in the set; deal ids are str")
    return set(found)


def checked_search(module, path: pathlib.Path):
    """module.search, run with the contract check; its exceptions become UserCodeError."""
    def run(ds: lab.Dataset, question: str, k: int = 10) -> list[int]:
        try:
            rows = module.search(ds, question, k=k)
        except (lab.LabError, lab.ModelError, labvespa.VespaError):
            raise
        except (Exception, SystemExit) as e:
            raise UserCodeError(e, path) from None
        try:
            return check_rows(rows, len(ds.ids), k)
        except ContractError as e:
            raise Fail(f"{e}\n  question: {question}") from None
    return run


def checked_answer(module, path: pathlib.Path):
    def run(question: str) -> set[str]:
        try:
            found = module.answer(question)
        except (lab.LabError, lab.ModelError):
            raise
        except (Exception, SystemExit) as e:
            raise UserCodeError(e, path) from None
        try:
            return check_deals(found)
        except ContractError as e:
            raise Fail(f"{e}\n  question: {question}") from None
    return run


# ---- eval history

def history() -> list[dict]:
    """The earlier evals on this version of the data. A line that is not one (cut short by a full disk or an
    interrupted write, or edited by hand) is skipped."""
    if not EVALS.exists():
        return []
    rows = []
    for line in EVALS.read_text(errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if (isinstance(r, dict) and all(isinstance(r.get(k), str) for k in ("file", "kind", "dataset"))
                and all(isinstance(r.get(k), (int, float)) for k in ("at", "score"))
                and r.get("data", lab.DATA_VERSION) == lab.DATA_VERSION):
            rows.append(r)
    return rows


def previous(past: list[dict], kind: str, dataset: str, file: str) -> dict | None:
    """The latest eval of file on dataset."""
    return max((r for r in past if (r["kind"], r["dataset"], r["file"]) == (kind, dataset, file)),
               key=lambda r: r["at"], default=None)


def best(past: list[dict], kind: str, dataset: str) -> dict | None:
    """The highest eval of any file on dataset; the earliest of equal ones."""
    return max((r for r in past if (r["kind"], r["dataset"]) == (kind, dataset)), key=lambda r: r["score"], default=None)


def flips(before: dict | None, per: dict[str, float]) -> tuple[int, int] | None:
    """How many questions of per scored higher and lower than in the eval before; None when that eval has no
    per-question scores."""
    old = before.get("per") if before else None
    if not isinstance(old, dict):
        return None
    pairs = [(old[q], s) for q, s in per.items() if isinstance(old.get(q), (int, float))]
    return sum(s > o for o, s in pairs), sum(s < o for o, s in pairs)


def recent_runs(past: list[dict], kind: str, n: int) -> list[tuple[dict, dict[str, float]]]:
    """The last n runs of kind, oldest first: one of its evals, and its score per dataset."""
    runs: dict[tuple, tuple[dict, dict[str, float]]] = {}
    for r in past:
        if r["kind"] == kind:  # a run's datasets share "run"; older lines share their file and second
            runs.setdefault((r["file"], r.get("run", int(r["at"]))), (r, {}))[1][r["dataset"]] = r["score"]
    return sorted(runs.values(), key=lambda run: run[0]["at"])[-n:]


SPARKS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    lo, hi = min(values), max(values)
    return "".join(SPARKS[round((v - lo) / (hi - lo) * 7) if hi > lo else 3] for v in values)


def remember(path: pathlib.Path, kind: str, run: float, result: Result) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    with open(EVALS, "a") as f:
        f.write(json.dumps(dict(at=time.time(), run=run, file=str(path), kind=kind, data=lab.DATA_VERSION,
                                dataset=result.dataset, score=result.score, lo=result.lo, hi=result.hi,
                                questions=len(result.per), per=result.per)) + "\n")


def progress(*extra) -> Progress:
    return Progress(SpinnerColumn(style="acc"), TextColumn("{task.description}"), BarColumn(complete_style="acc"), *extra,
                    TimeElapsedColumn(), console=console, transient=True)


# ---- the eval screen

@dataclass
class Result:
    dataset: str
    per: dict[str, float]  # question -> its score, rounded as evals.jsonl keeps it
    score: float
    lo: float
    hi: float


def result(dataset: str, per: dict[str, float]) -> Result:
    x = np.array(list(per.values()), dtype=float)
    lo, hi = lab.bootstrap(x)
    return Result(dataset, {q: round(float(s), 4) for q, s in per.items()}, float(x.mean()), lo, hi)


def when(at: float) -> str:
    """The time of day, with the weekday unless it is today."""
    t = time.localtime(at)
    return time.strftime("%H:%M" if t[:3] == time.localtime()[:3] else "%a %H:%M", t)


def short(file: str, width: int = 16) -> str:
    name = pathlib.Path(file).name
    return name if len(name) <= width else name[:width - 1] + "…"


def earlier(r: dict | None, file: bool = True) -> Text:
    """An earlier eval's score, time and file, or a dash."""
    if r is None:
        return Text("–", style="dim")
    return Text.assemble(f"{r['score']:.3f}", (f" {when(r['at'])}" + (f" {short(r['file'])}" if file else ""), "dim"))


def result_table(path: pathlib.Path, kind: str, metric: str, results: list[Result], past: list[dict],
                 elapsed: float, caption: str = "") -> Table:
    """This run's scores, each next to the previous eval of the same file and the best earlier eval of any file."""
    cells, record = [], False
    for r in results:
        before, top = previous(past, kind, r.dataset, str(path)), best(past, kind, r.dataset)
        moved = flips(before, r.per)
        new_best = top is not None and r.score > top["score"]
        record |= new_best
        cells.append([Text(r.dataset), Text.assemble((f"{r.score:.3f}", "num"), (" ★", "ok") if new_best else "  "),
                      Text(f"{r.lo:.3f}–{r.hi:.3f}"),
                      Text.assemble((f"↑{moved[0]}", "ok"), " ", (f"↓{moved[1]}", "err")) if moved else Text("–", "dim"),
                      earlier(before, file=False), earlier(top)])
    table = Table(title=f"{path.name} · {metric} · {elapsed:.1f} s", title_justify="left", title_style="bold",
                  box=None, padding=(0, 1), caption_style="dim", caption_justify="left")
    for i, (header, justify) in enumerate(((("dataset", "left"), (metric, "right"), ("95% interval", "right"),
                                            ("↑ ↓", "right"), ("previous", "left"), ("best before", "left")))):
        # in a narrow terminal only the last column gives way, cutting its file name short
        width = max([len(header)] + [c[i].cell_len for c in cells]) if i < 5 else None
        table.add_column(header, justify=justify, header_style="dim", no_wrap=True, overflow="ellipsis", min_width=width)
    for c in cells:
        table.add_row(*c)
    notes = [f"previous: the last eval of {path.name} · ↑ ↓ questions now higher, lower"]
    if record:
        notes.append("★ above every earlier eval")
    table.caption = "\n".join(([caption] if caption else []) + notes)
    return table


def runs_table(kind: str, datasets: list[str], past: list[dict], n: int = 5, trend: int = 20) -> Table | None:
    """The last n earlier runs of kind, oldest first, and a sparkline of each dataset over the last trend runs."""
    runs = recent_runs(past, kind, trend)
    if not runs:
        return None
    table = Table(title="earlier evals", title_justify="left", title_style="dim", box=None, padding=(0, 1))
    table.add_column("", style="dim", no_wrap=True)
    table.add_column("file", no_wrap=True, overflow="ellipsis", max_width=24)
    for d in datasets:
        table.add_column(d, justify="right", header_style="dim", no_wrap=True)
    for first, scores in runs[-n:]:
        table.add_row(when(first["at"]), short(first["file"], 24),
                      *(f"{scores[d]:.3f}" if d in scores else Text("–", "dim") for d in datasets))
    lines = [[scores[d] for _, scores in runs if d in scores] for d in datasets]
    if any(len(v) > 1 for v in lines):
        table.add_row(f"last {len(runs)}", "", *(Text(sparkline(v), "acc") if len(v) > 1 else "" for v in lines))
    return table


def show_end(*renderables) -> None:
    for r in renderables:
        if r is not None:
            console.print()
            console.print(r)


def quiet_models() -> None:
    """No model-loading bars or Hugging Face notices among the eval's output; the CLI reports cache misses itself."""
    for name, value in (("HF_HUB_DISABLE_PROGRESS_BARS", "1"), ("TRANSFORMERS_VERBOSITY", "error"),
                        ("HF_HUB_VERBOSITY", "error"), ("TOKENIZERS_PARALLELISM", "false")):
        os.environ.setdefault(name, value)
    warnings.simplefilter("ignore", lab.RerankCacheMiss)


# ---- eval

def eval_search(path: pathlib.Path, module, worst: int, only: str | None) -> None:
    search = checked_search(module, path)
    run = time.time()
    with console.status("loading the data"):
        sets = [(f"{name} {split}", lab.load(name, split)) for name, split in EVAL_SETS if only in (None, name)]
    scores, ranked = {}, {}
    with progress(MofNCompleteColumn()) as bar:
        tasks = {label: bar.add_task(label, total=len(ds.qids)) for label, ds in sets}
        for label, ds in sets:
            for q in ds.qids:
                rows = [r for r in search(ds, ds.queries[q], k=10) if ds.ids[r] not in ds.hidden[q]]
                ranked[label, q] = rows
                scores[label, q] = lab.ndcg10([ds.ids[i] for i in rows], ds.qrels[q])
                bar.advance(tasks[label])
    elapsed = time.time() - run
    results = [result(label, {q: scores[label, q] for q in ds.qids}) for label, ds in sets]
    past = history()
    if worst:
        show_worst_search(sets, scores, ranked, worst)
    show_end(runs_table("search", EVAL_LABELS, past), result_table(path, "search", "nDCG@10", results, past, elapsed))
    if lab.rerank_misses[0]:
        console.print(f"[warn]![/] {lab.rerank_misses[0]:,} reranker pairs missed the cache and scored lowest (no GPU to "
                      "score them)")
    for r in results:
        remember(path, "search", run, r)


def show_worst_search(sets, scores: dict, ranked: dict, n: int) -> None:
    """The n worst questions: your top 10 with their grades (2 and 1 relevant, . not), then relevant documents it missed."""
    by_label = dict(sets)
    for label, q in sorted(scores, key=scores.get)[:n]:
        ds = by_label[label]
        grades = ds.qrels[q]
        console.print(f"\n[num]{scores[label, q]:.3f}[/]  [dim]{label} · question {q}[/]")
        console.print(f"  {ds.queries[q]}", markup=False)
        for rank, i in enumerate(ranked[label, q], 1):
            doc_line(ds, f"{rank:>4}.", i, grades.get(ds.ids[i], 0))
        found = {ds.ids[i] for i in ranked[label, q]}
        for d, g in sorted(grades.items(), key=lambda x: -x[1]):
            if g > 0 and d not in found and d in ds.row and d not in ds.hidden[q]:
                doc_line(ds, "  missed", ds.row[d], g)


def doc_line(ds: lab.Dataset, prefix: str, i: int, grade: int) -> None:
    mark = {2: ("2", "grade2"), 1: ("1", "ok")}.get(grade, (".", "dim"))
    text = " ".join(f"{ds.titles[i]} {ds.bodies[i]}".split())[:400]
    console.print(Text.assemble(f"{prefix} [", mark, f"] {ds.ids[i]:<16} ", (text, "" if grade else "dim")),
                  no_wrap=True, overflow="ellipsis")


AGENT_SET = "crm dev"


def eval_agent(path: pathlib.Path, module, worst: int) -> None:
    with contextlib.redirect_stderr(io.StringIO()):  # the model is named in the caption below
        url, _, model = require_agent()
    where = "on this machine" if url.startswith(lab.VLLM_URL) else "through the lab"
    answer = checked_answer(module, path)
    questions = lab.crm_dev()
    found: dict[int, set[str]] = {}
    run = time.time()
    pool = ThreadPoolExecutor(lab.AGENT_THREADS)
    try:
        with progress(MofNCompleteColumn()) as bar:
            task = bar.add_task(AGENT_SET, total=len(questions))
            futures = {pool.submit(answer, q): i for i, (q, _) in enumerate(questions)}
            for f in as_completed(futures):
                found[futures[f]] = f.result()
                bar.advance(task)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # an error or Ctrl-C does not wait for the running agents
    elapsed = time.time() - run
    res = result(AGENT_SET, {q: float(found[i] == gold) for i, (q, gold) in enumerate(questions)})
    past = history()
    for i in [i for i, (q, _) in enumerate(questions) if not res.per[q]][:worst]:
        q, gold = questions[i]
        console.print(Text.assemble("\n", ("✗ ", "err"), q))
        console.print(f"  [dim]expected[/] {', '.join(sorted(gold)) or 'none'}")
        console.print(f"  [dim]missing [/] {', '.join(sorted(gold - found[i])) or '–'}")
        console.print(f"  [dim]extra   [/] {', '.join(sorted(found[i] - gold)) or '–'}")
    last = (previous(past, "agent", AGENT_SET, str(path)) or {}).get("per")
    console.print()
    for q, _ in questions:
        was = last.get(q) if isinstance(last, dict) else None
        moved = ("↑ ", "ok") if was == 0 and res.per[q] else ("↓ ", "err") if was == 1 and not res.per[q] else "  "
        console.print(Text.assemble("  ", ("✓ ", "ok") if res.per[q] else ("✗ ", "err"), moved, q),
                      no_wrap=True, overflow="ellipsis")
    show_end(runs_table("agent", [AGENT_SET], past),
             result_table(path, "agent", "exact", [res], past, elapsed, f"agent model {model} {where}"))
    remember(path, "agent", run, res)


def cmd_eval(args) -> None:
    quiet_models()
    path, module = load_file(args.file)
    if file_kind(module) == "search":
        eval_search(path, module, args.worst, args.data)
    elif args.data:
        raise Fail("--data picks a search dataset; an agent file runs on the CRM questions")
    else:
        eval_agent(path, module, args.worst)


# ---- submit

def cmd_submit(args) -> None:
    label = args.m.strip()
    if not 1 <= len(label) <= LABEL_CHARS:
        raise Fail(f"-m: a label of 1 to {LABEL_CHARS} characters on what your file does (yours has {len(label)})")
    quiet_models()
    path, module = load_file(args.file)
    kind = file_kind(module)
    with progress(TextColumn("{task.completed} questions")) as bar:
        task = bar.add_task("checking on one dev question", total=None)
        if kind == "search":
            ds, search = lab.load(BOARD_DATASET), checked_search(module, path)
            search(ds, ds.queries[ds.qids[0]], k=10)
            bar.update(task, description="running on the board's questions")
            result = lab.submit(ds, lambda ds_, question, k=10: counted(bar, task, search(ds_, question, k)), label)
        else:
            require_agent()
            answer = checked_answer(module, path)
            answer(lab.CRM_QUESTIONS[0][0])
            bar.update(task, description="running on the board's questions")
            result = lab.submit_agent(lambda question: counted(bar, task, answer(question)), label)
    rank = f" · rank [num]{result['rank']}[/] of {result['of']}" if "rank" in result and "of" in result else ""
    ok(f"submitted {path.name} · score [num]{result['score']:.4f}[/] · your best {result['best']:.4f}{rank}")


def counted(bar: Progress, task, result):
    bar.advance(task)
    return result


# ---- board

def board_view(kind: str, me: str | None) -> Table:
    data = lab.call("/api/board" if kind == "search" else "/api/agent-board")
    what = "FiQA nDCG@10" if kind == "search" else "CRM exact set"
    table = Table(title=f"{kind} board · {what} · {data['questions']} questions", title_justify="left", title_style="bold",
                  box=None, padding=(0, 2), caption=time.strftime("updated %H:%M:%S"), caption_style="dim",
                  caption_justify="left")
    for col, just in (("#", "right"), ("nick", "left"), ("score", "right"), ("label", "left"), ("subs", "right"),
                      ("at", "right")):
        table.add_column(col, justify=just, header_style="dim")
    for r in data["rows"]:
        style = "me" if r["nick"] == me else ("bold" if r["rank"] == 1 else "")
        table.add_row(str(r["rank"]), r["nick"], f"{r['score']:.4f}", Text(r["label"]), str(r["submissions"]),
                      time.strftime("%a %H:%M", time.localtime(r["at"])), style=style)
    if not data["rows"]:
        table.add_row("", "[dim]no submissions yet[/]", "", "", "", "")
    return table


def cmd_board(args) -> None:
    try:
        me = lab.call("/api/me")["nick"]
    except lab.LabError:
        me = None
    if not args.watch:
        console.print(board_view(args.kind, me))
        return
    last = board_view(args.kind, me)
    with Live(last, console=console, refresh_per_second=1) as live:
        try:
            while True:
                time.sleep(10)
                try:
                    live.update(last := board_view(args.kind, me))
                except (lab.LabError, OSError) as e:
                    live.update(Group(last, Text(f"✗ cannot reach the lab ({e}); retrying every 10 s", style="err")))
        except KeyboardInterrupt:
            pass


# ---- setup, serve, status

def gpu() -> tuple[str, int, int] | None:
    """(name, free bytes, total bytes) of GPU 0, or None without one."""
    import torch

    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info(0)
    return torch.cuda.get_device_name(0), free, total


def vllm_installed() -> bool:
    python = VENV / "bin" / "python"
    if not python.exists():
        return False
    r = subprocess.run([str(python), "-c", "import vllm; print(vllm.__version__)"], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == lab.VLLM_VERSION


def install_vllm() -> None:
    """vLLM in its own venv, so its torch pin does not touch the lab's packages."""
    HOME.mkdir(parents=True, exist_ok=True)
    log = HOME / "vllm-install.log"
    with open(log, "w") as out, console.status(f"installing vLLM {lab.VLLM_VERSION} into {VENV} (a few minutes)") as status:
        steps = ([] if (VENV / "bin" / "python").exists() else [[sys.executable, "-m", "venv", str(VENV)]]) + [
            [str(VENV / "bin" / "pip"), "install", "--upgrade", "pip"],
            [str(VENV / "bin" / "pip"), "install", f"vllm=={lab.VLLM_VERSION}"]]
        for cmd in steps:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                out.write(line)
                if line.strip():
                    status.update(f"installing vLLM {lab.VLLM_VERSION}  [dim]{line.strip()[:80]}[/]")
            if proc.wait():
                raise Fail(f"installing vLLM failed; see {log}")


LAPTOP = "laptop mode: no GPU; reranker from the precomputed cache, agent through the lab server"


def cmd_setup(args) -> None:
    with console.status("checking the GPU"):
        found = gpu()
    if found is None:
        console.print(f"[warn]![/] {LAPTOP}")
    else:
        name, free, total = found
        if free < MIN_FREE_GB * 1e9 and not args.force:
            raise Fail(f"{name} has {gb(free)} free of {gb(total)}; the lab needs {MIN_FREE_GB} GB free "
                       f"(an 80 GB GPU such as an A100 80GB or RTX PRO 6000). Run on a larger GPU, or pass --force to try anyway")
        ok(f"GPU  {name} · {gb(free)} free of {gb(total)}")

    download("data", lab.DATA / lab.DATA_VERSION, lab.fetch)
    cache = download("reranker scores", lab.DATA / lab.DATA_VERSION / lab.rerank_cache_name(), lab.fetch_rerank_cache)
    if cache is None:
        console.print(f"[warn]![/] the release has no {lab.rerank_cache_name()} yet; "
                      + ("the reranker scores on the GPU" if found else "unscored pairs rank lowest"))
    with console.status("building the search indexes (once; lab eval reads them from data/cache)"):
        for name, split in EVAL_SETS:
            ds = lab.load(name, split)
            lab.hybrid(ds, ds.queries[ds.qids[0]])
    ok("search indexes")

    from huggingface_hub import snapshot_download

    models = [("encoder", lab.QUERY_MODEL, lab.QUERY_REVISION)]
    if found:
        models += [("reranker", lab.RERANK_MODEL, None), ("agent model", lab.AGENT_MODEL, None)]
    for what, repo, rev in models:
        console.print(f"[acc]→[/] {what}  {repo}")
        snapshot_download(repo, revision=rev)
        ok(f"{what}  {repo}")

    if found is None:
        console.print("\n[dim]next:[/] lab connect URL TOKEN [dim](both on your lab page)[/]")
        return
    if vllm_installed():
        ok(f"vLLM {lab.VLLM_VERSION}")
    else:
        install_vllm()
        ok(f"vLLM {lab.VLLM_VERSION}  {VENV}")
    console.print("\n[dim]next:[/] lab connect URL TOKEN [dim](both on your lab page), then[/] lab serve")


def download(what: str, path: pathlib.Path, fetch) -> pathlib.Path | None:
    """fetch(progress) with a progress bar, unless path is there already."""
    if not path.exists():
        with Progress(TextColumn(f"  {what}"), BarColumn(complete_style="acc"), DownloadColumn(), TransferSpeedColumn(),
                      console=console, transient=True) as bar:
            task = bar.add_task(what, total=None)
            if fetch(progress=lambda done, total: bar.update(task, completed=done, total=total or None)) is None:
                return None
    ok(f"{what}  {path.name}")
    return path


def require_agent() -> tuple[str, dict[str, str], str]:
    """Where chat sends its messages (lab.agent_endpoint); fails unless it has a model: the local vLLM, or the lab
    server."""
    try:
        return lab.agent_endpoint()
    except RuntimeError as e:
        raise Fail(str(e)) from None


def serve_pid() -> int | None:
    try:
        pid = int(SERVE_PID.read_text())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def tail(path: pathlib.Path, n: int = 15) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:]) if path.exists() else ""


def vllm_args() -> list[str]:
    """lab.VLLM_ARGS with the model's download path filled in, and a GPU memory share that leaves
    lab.VLLM_LEAVE_GB free for the reranker and the encoder."""
    import torch
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(lab.AGENT_MODEL, local_files_only=True)
    except Exception:  # not downloaded (huggingface_hub raises several types for this)
        raise Fail(f"{lab.AGENT_MODEL} is not downloaded; run lab setup first") from None
    total = torch.cuda.mem_get_info(0)[1]
    if total < MIN_FREE_GB * 1e9:
        raise Fail(f"the GPU has {gb(total)}; {lab.AGENT_MODEL} and the reranker need a card of {MIN_FREE_GB} GB or more")
    share = round(1 - lab.VLLM_LEAVE_GB * 1e9 / total, 2)
    return [a.replace("{snapshot}", snapshot) for a in lab.VLLM_ARGS] + ["--gpu-memory-utilization", str(share)]


def cmd_serve(args) -> None:
    if args.stop:
        stop_vllm()
        return
    if (model := lab.local_agent_model()) is not None:
        if model != lab.AGENT_MODEL:
            raise Fail(f"{lab.VLLM_URL} serves {model}, not {lab.AGENT_MODEL}; stop that server, then run lab serve")
        ok(f"vLLM is already up · {model}")
        return
    proc = None
    if serve_pid() is None:
        binary = VENV / "bin" / "vllm"
        if not binary.exists():
            raise Fail("vLLM is not installed; run lab setup first (on a GPU machine; without one, chat goes "
                       "through the lab server and there is nothing to serve)")
        HOME.mkdir(parents=True, exist_ok=True)
        with open(SERVE_LOG, "w") as log:
            proc = subprocess.Popen([str(binary), "serve", lab.AGENT_MODEL, "--host", "127.0.0.1",
                                     "--port", str(urllib.parse.urlsplit(lab.VLLM_URL).port),
                                     *vllm_args()], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
        SERVE_PID.write_text(str(proc.pid))
    start, timeout = time.time(), args.timeout
    try:
        with console.status("") as status:
            while lab.local_agent_model() is None:
                if not starting(proc):
                    SERVE_PID.unlink(missing_ok=True)
                    console.print(Panel(Text(tail(SERVE_LOG), style="dim"), title="serve.log", title_align="left",
                                        border_style="err"))
                    raise Fail(f"vLLM stopped while starting; the full log is {SERVE_LOG}")
                if time.time() - start > timeout:
                    raise Fail(f"vLLM did not answer within {timeout} s; it keeps starting in the background. "
                               f"Look at {SERVE_LOG}, or stop it with lab serve --stop")
                last = tail(SERVE_LOG, 1)[:90]
                status.update(f"starting {lab.AGENT_MODEL} on the GPU · {time.time() - start:.0f} s  [dim]{last}[/]")
                time.sleep(2)
    except KeyboardInterrupt:
        console.print("[dim]vLLM keeps starting in the background; lab status shows when it is up[/]")
        return
    ok(f"vLLM is up · {lab.AGENT_MODEL} at {lab.VLLM_URL} · log {SERVE_LOG}")


def starting(proc: subprocess.Popen | None) -> bool:
    """Whether vLLM still runs: proc when this command started it (a child that exited stays a zombie, which
    os.kill still finds), else the process in SERVE_PID."""
    return proc.poll() is None if proc is not None else serve_pid() is not None


def stop_vllm() -> None:
    pid = serve_pid()
    if pid is None:
        SERVE_PID.unlink(missing_ok=True)
        ok("vLLM is not running")
        return
    os.killpg(pid, signal.SIGTERM)
    for _ in range(60):
        if serve_pid() is None:
            break
        time.sleep(0.5)
    else:
        os.killpg(pid, signal.SIGKILL)
    SERVE_PID.unlink(missing_ok=True)
    ok("vLLM stopped")


def cmd_status(args) -> None:
    from huggingface_hub import snapshot_download

    def cached(repo: str, rev: str | None = None) -> bool:
        try:
            snapshot_download(repo, revision=rev, local_files_only=True)
            return True
        except Exception:  # huggingface_hub raises several kinds for a missing snapshot
            return False

    with console.status("checking"):
        found = gpu()
        model = lab.local_agent_model()
        try:
            nick = lab.call("/api/me")["nick"] if lab.LAB.exists() else None
            conn = f"[ok]connected[/] as [num]{nick}[/] · {json.loads(lab.LAB.read_text())['url']}" if nick else \
                "[warn]not connected[/] · lab connect URL TOKEN"
        except lab.LabError as e:
            conn = f"[err]refused[/] · {e}"
        except OSError as e:
            conn = f"[err]unreachable[/] · {getattr(e, 'reason', e)}"
        needed = [("encoder", lab.QUERY_MODEL, lab.QUERY_REVISION)] + (
            [("reranker", lab.RERANK_MODEL, None), ("agent", lab.AGENT_MODEL, None)] if found else [])
        models = " · ".join(f"{what} {'[ok]✓[/]' if cached(repo, rev) else '[err]✗[/]'}" for what, repo, rev in needed)
        scores = (lab.DATA / lab.DATA_VERSION / lab.rerank_cache_name()).exists()
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column()
    grid.add_row("mode", "GPU" if found else f"[warn]{LAPTOP.removeprefix('laptop mode: ')}[/]")
    grid.add_row("GPU", f"{found[0]} · {gb(found[1])} free of {gb(found[2])}" if found else "[warn]none[/]")
    grid.add_row("agent", f"[ok]{model}[/] on this machine" if model == lab.AGENT_MODEL else
                 f"{lab.LAB_AGENT_MODEL} through the lab · lab serve runs {lab.AGENT_MODEL} here" if found else
                 f"{lab.LAB_AGENT_MODEL} through the lab")
    grid.add_row("lab", conn)
    grid.add_row("data", f"{lab.DATA_VERSION} [ok]✓[/]" if (lab.DATA / lab.DATA_VERSION).exists() else "[err]missing[/] · lab setup")
    grid.add_row("models", models)
    grid.add_row("reranker scores", f"{lab.rerank_cache_name()} " + ("[ok]✓[/]" if scores else "[err]✗[/] · lab setup"))
    console.print(Panel(grid, title="lab status", title_align="left", border_style="acc", expand=False))


# ---- connect, new

def cmd_connect(args) -> None:
    ok(f"connected as [num]{lab.connect(args.url, args.token)}[/]")


def cmd_new(args) -> None:
    if args.kind == "vespa":
        new_vespa(pathlib.Path.cwd() / "vespa")
        return
    target = pathlib.Path.cwd() / f"{args.kind}.py"
    try:
        with open(target, "x") as f:
            f.write((TEMPLATES / f"{args.kind}.py").read_text())
    except FileExistsError:
        raise Fail(f"{target.name} exists; rename or remove it first") from None
    except OSError as e:
        raise Fail(f"cannot write {target}: {e.strerror}; run lab new in a directory of yours") from None
    ok(f"wrote {target.name} · next: lab eval {target.name}")


NICK_FILE = "# your nick on the lab board: the organizer grades this branch under it\n{nick}\n"


def new_vespa(target: pathlib.Path) -> None:
    """target/app (the lab's Vespa application), target/search.py and target/NICK, each unless it is there already."""
    wrote, kept = [], []
    try:
        target.mkdir(exist_ok=True)
        if (target / "app").exists():
            kept.append("app")
        else:
            shutil.copytree(labvespa.APP, target / "app")
            wrote.append("app")
        for name, text in (("search.py", lambda: (TEMPLATES / "vespa" / "search.py").read_text()),
                           ("NICK", lambda: NICK_FILE.format(nick=my_nick() or "your-nick"))):
            if (target / name).exists():
                kept.append(name)
            else:
                (target / name).write_text(text())
                wrote.append(name)
    except OSError as e:
        raise Fail(f"cannot write {target}: {e.strerror}; run lab new in a directory of yours") from None
    if not wrote:
        raise Fail(f"{target.name}/ has app, search.py and NICK already; rename or remove what you want new")
    ok(f"wrote {', '.join(f'{target.name}/{w}' for w in wrote)}" + (f" · kept {', '.join(kept)}" if kept else "")
       + f" · next: lab vespa up, then lab eval {target.name}/search.py")


def my_nick() -> str | None:
    """The nick of the connected participant, or None when not connected."""
    try:
        return lab.call("/api/me")["nick"] if lab.LAB.exists() else None
    except (RuntimeError, OSError, KeyError, ValueError):  # lab.LabError, an unreachable lab, a reply without a nick
        return None


# ---- vespa

def cmd_vespa(args) -> None:
    if args.app and args.action in ("down", "status"):
        raise Fail(f"lab vespa {args.action} takes no application directory")
    {"up": vespa_up, "deploy": vespa_deploy, "down": vespa_down, "status": vespa_status}[args.action](args.app)


def vespa_up(app: str | None) -> None:
    start = time.time()
    with progress(MofNCompleteColumn()) as bar:
        step = bar.add_task("starting Vespa", total=None)

        def feeding(name: str, total: int):
            bar.update(step, visible=False)
            task = bar.add_task(f"feeding {name}", total=total)
            return lambda n: bar.advance(task, n)

        result = labvespa.up(app, lambda m: bar.update(step, description=m), feeding, warn)
    show_changes(result["deploy"])
    fed = " · ".join(f"fed {name} in {s:.0f} s" for name, s in result["fed"].items()) or "the data was fed before"
    ok(f"Vespa is up at {labvespa.QUERY_URL} · {fed} · {time.time() - start:.0f} s in all")


def vespa_deploy(app: str | None) -> None:
    path = labvespa.app_dir(app)
    with console.status("deploying") as status:
        answer = labvespa.deploy(path, lambda m: status.update(m), warn)
    show_changes(answer)
    ok(f"deployed {path}")
    unfed = labvespa.unfed()
    if unfed:
        warn(f"{' and '.join(unfed)}: {'; '.join(sorted(set(unfed.values())))}; lab vespa up {app or ''}".rstrip()
             + " feeds them (about 1.5 minutes)")


def show_changes(answer: dict) -> None:
    """Vespa's warnings on a deploy, and what it needs restarted, refed or reindexed."""
    lines = labvespa.changes(answer)
    for line in lines:
        warn(line)
    if any(line.startswith("restart") for line in lines):
        warn(f"to restart: docker restart {labvespa.NAME}, then lab vespa up")
    if any(line.startswith("reindex") for line in lines):
        warn("Vespa reindexes the documents in the background")


def warn(message: str) -> None:
    console.print(Text.assemble(("! ", "warn"), message))


def vespa_down(app: str | None) -> None:
    ok(f"removed {labvespa.NAME} and its data" if labvespa.down() else f"no {labvespa.NAME} container")


def vespa_status(app: str | None) -> None:
    with console.status("checking"):
        s = labvespa.status()
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column()
    grid.add_row("container", f"{labvespa.NAME} · {s['image']} · " + (
        f"[ok]{s['state']}[/]" if s["state"] == "running" else f"[warn]{s['state'] or 'none'}[/] · lab vespa up"))
    if s["state"] == "running":
        grid.add_row("memory", s["memory"])
        grid.add_row("queries", f"{s['query']} · " + ("[ok]serving[/]" if s["serving"] else "[warn]not serving[/] · lab vespa up"))
        for name, d in s["datasets"].items():
            grid.add_row(name, f"{d['documents'] if d['documents'] is not None else '–'} documents · " + (
                f"[warn]{d['problem']}[/] · lab vespa up" if d["problem"] else "[ok]fed[/]"))
    console.print(Panel(grid, title="lab vespa status", title_align="left", border_style="acc", expand=False))


# ---- main

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lab", description=__doc__)
    sub = p.add_subparsers(required=True, metavar="command")
    s = sub.add_parser("setup", help="check the GPU; download the data and models; install vLLM")
    s.add_argument("--force", action="store_true", help=f"continue with less than {MIN_FREE_GB} GB of free GPU memory")
    s.set_defaults(run=cmd_setup)
    s = sub.add_parser("connect", help="save the lab URL and your token (both on your lab page)")
    s.add_argument("url")
    s.add_argument("token")
    s.set_defaults(run=cmd_connect)
    s = sub.add_parser("serve", help="start the agent model (vLLM) in the background")
    s.add_argument("--stop", action="store_true", help="stop it")
    s.add_argument("--timeout", type=int, default=900, help="seconds to wait for it (default 900)")
    s.set_defaults(run=cmd_serve)
    s = sub.add_parser("status", help="GPU, vLLM, lab connection, data and models")
    s.set_defaults(run=cmd_status)
    s = sub.add_parser("new", help="copy a working template into this directory")
    s.add_argument("kind", choices=("search", "agent", "vespa"))
    s.set_defaults(run=cmd_new)
    s = sub.add_parser("vespa", help="your own Vespa in Docker: up, deploy, down, status")
    s.add_argument("action", choices=("up", "deploy", "down", "status"))
    s.add_argument("app", nargs="?", help="the application directory (default: ./vespa/app, else the lab's)")
    s.set_defaults(run=cmd_vespa)
    s = sub.add_parser("eval", help="score your file on the dev questions")
    s.add_argument("file")
    s.add_argument("--worst", type=int, default=0, metavar="N", help="also show the N worst questions")
    s.add_argument("--data", choices=[name for name, _ in EVAL_SETS], help="only this dataset (search files)")
    s.set_defaults(run=cmd_eval)
    s = sub.add_parser("submit", help="run your file on the board's questions and send the results")
    s.add_argument("file")
    s.add_argument("-m", required=True, metavar="LABEL", help="a few words on what it does")
    s.set_defaults(run=cmd_submit)
    s = sub.add_parser("board", help="the leaderboard")
    s.add_argument("kind", nargs="?", choices=("search", "agent"), default="search")
    s.add_argument("--watch", action="store_true", help="refresh every 10 s")
    s.set_defaults(run=cmd_board)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        args.run(args)
        return 0
    except UserCodeError as e:
        show_user_error(e)
    except Fail as e:
        fail(str(e))
    except lab.LabError as e:
        fail(f"the lab says: {e.detail}" + (" · run lab connect URL TOKEN again" if e.status == 401 else ""))
    except urllib.error.URLError as e:
        fail(f"cannot reach the lab: {e.reason}")
    except RuntimeError as e:  # lab's own messages: not connected, the agent model is down
        fail(str(e))
    except OSError as e:  # the CLI's own files: an unwritable directory, a full disk
        fail(str(e))
    except KeyboardInterrupt:
        console.print("[dim]stopped[/]")
        if threading.active_count() > 1:  # agent threads can wait minutes on the model; exit without joining them
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(130)
        return 130
    return 1


if __name__ == "__main__":
    sys.exit(main())
