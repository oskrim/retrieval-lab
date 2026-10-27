"""The lab CLI: set up the GPU box, evaluate and submit your search/ folder or agent.py."""

from __future__ import annotations

import argparse
import contextlib
import functools
import gzip
import hashlib
import html
import importlib.util
import io
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from rich.console import Console, Group
from rich.markup import escape
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
BIG_GPU_GB = 60
LABEL_CHARS = 60  # the lab's limit on a submission's label
EVAL_SETS = (("fiqa", "dev"), ("caiq", "test"))  # caiq has no dev split
EVAL_LABELS = [f"{name} {split}" for name, split in EVAL_SETS]
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


def named_file(word: str) -> str:
    """The file a command means: search is search/search.py; agent is agent.py; anything else a path."""
    if word == "search":
        if not (labvespa.FOLDER / "search.py").is_file():
            raise Fail(f"no {labvespa.FOLDER}/search.py here; {elsewhere(labvespa.FOLDER)}")
        return str(labvespa.FOLDER / "search.py")
    if word == "agent" and not pathlib.Path("agent.py").is_file():
        raise Fail(f"no agent.py here; {elsewhere(pathlib.Path('agent.py'))}")
    return "agent.py" if word == "agent" else word


def elsewhere(name: pathlib.Path) -> str:
    """Where the lab's own name is: run lab in the lab's folder, or restore it there."""
    if pathlib.Path.cwd().resolve() == lab.HERE:
        return f"git checkout -- {name.as_posix()} brings back the lab's"
    return f"run lab in {lab.HERE}, which has the lab's"


def file_kind(module) -> str:
    """"agent" for a file with answer() (its tools may include one named search), else "search" for one with
    search()."""
    if callable(getattr(module, "answer", None)):
        return "agent"
    if callable(getattr(module, "search", None)):
        return "search"
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


# ---- evals.jsonl: one line per dataset of each eval

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
            if not (isinstance(r.get("name"), str) and isinstance(r.get("id"), str)):
                r.pop("name", None)
                r.pop("id", None)
            rows.append(r)
    return rows


@functools.lru_cache(maxsize=None)
def file_key(file: str) -> str:
    """The file's path in its git repository (search/search.py, my.py), else its whole path: evals compare files by
    it, so search/search.py and other files stay apart and the same file in another clone is the same."""
    path = pathlib.Path(file)
    for parent in path.parents:
        if (parent / ".git").exists():
            return path.relative_to(parent).as_posix()
    return str(path)


def remember(path: pathlib.Path, kind: str, run: float, result: Result, run_id: str, name: str) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    with open(EVALS, "a") as f:
        f.write(json.dumps(dict(at=time.time(), run=run, id=run_id, name=name, file=str(path), kind=kind,
                                data=lab.DATA_VERSION, dataset=result.dataset, score=result.score, lo=result.lo,
                                hi=result.hi, questions=len(result.per))) + "\n")


# ---- saved runs: each eval's scores and top documents per question, under a name like brave-otter-42

TOP = 20  # documents a saved run keeps per question; nDCG@10 scores the first 10
KEEP_RUNS = 200
RUN_FORMAT = 1
ADJECTIVES = """able amber azure bold brave brisk calm cheery clever cosmic cozy crafty crisp curious dandy dapper daring
eager early easy fancy fast fierce fluffy fond fresh frosty gentle giddy glad golden grand happy hardy honest humble icy
jazzy jolly keen kind lively loyal lucky lunar mellow merry mighty misty modest nifty nimble noble odd peppy perky plucky
polite proud quick quiet rapid rosy royal rusty salty sandy sharp shiny shy silent silly sleek sly smart snowy solar spicy
steady stormy sturdy sunny sweet swift tame tender tidy tiny true upbeat velvet vivid warm wild windy wise witty young
zesty zippy""".split()
NOUNS = """ant badger bat bear beaver bee bison boar camel carp cat clam cobra colt crab crane crow deer dingo dove duck
eagle egret elk emu falcon ferret finch fox frog gecko gibbon goose gull hare hawk heron hippo horse ibis impala jackal
jay kiwi koala lark lemur lion llama magpie marten mink mole moose mouse mule newt okapi orca osprey owl panda parrot
pelican pony puffin quail rabbit ram rat raven robin salmon seal shark sloth slug snail sparrow squid stork swan tapir
tern tiger toad trout turtle viper vole walrus weasel whale wolf wombat wren yak zebra""".split()
NAME = re.compile(r"[a-z]+-[a-z]+-[0-9a-f]+(#[0-9]+)?")  # #2, #3, ...: the later evals of an agent configuration


def config_id(path: pathlib.Path) -> str:
    """A digest of what decides the run's results: the data version, the file and, for the search.py of a search
    folder (with app/ and feed.py), every other file of the folder (feed.py, analyze.py, synonyms.json, app/...)."""
    parts = [("", path)]
    folder = path.parent
    if path.name == "search.py" and (folder / "app").is_dir() and (folder / "feed.py").is_file():
        parts += sorted((p.relative_to(folder).as_posix(), p) for p in folder.rglob("*")
                        if p.is_file() and p != path and p.suffix != ".pyc"
                        and not any(x.startswith(".") or x == "__pycache__" for x in p.relative_to(folder).parts))
    h = hashlib.sha256(lab.DATA_VERSION.encode())
    for name, p in parts:
        data = p.read_bytes()
        h.update(f"\0{name}\0{len(data)}\0".encode())
        h.update(data)
    return h.hexdigest()[:16]


def run_name(run_id: str, taken: dict[str, str]) -> str:
    """The run's name: adjective-noun-number from run_id, so the same configuration has the same name. When another
    run has that name already (taken: name -> id), the number counts on."""
    h = int(run_id, 16)
    adjective, noun = ADJECTIVES[h % len(ADJECTIVES)], NOUNS[h // len(ADJECTIVES) % len(NOUNS)]
    first = h // (len(ADJECTIVES) * len(NOUNS)) % 99
    for step in range(99):
        name = f"{adjective}-{noun}-{(first + step) % 99 + 1}"
        if taken.get(name, run_id) == run_id:
            return name
    return f"{adjective}-{noun}-{run_id[:8]}"


def base_name(name: str) -> str:
    """A run's name without its eval number: curious-parrot-60#2 is an eval of curious-parrot-60."""
    return name.partition("#")[0]


def taken_names(past: list[dict]) -> dict[str, str]:
    return {base_name(r["name"]): r["id"] for r in past if "name" in r}


def eval_name(name: str, kind: str, past: list[dict]) -> str:
    """The name an eval is saved under. A search's eval of the same configuration replaces the earlier one (name). An
    agent's answers vary from eval to eval, so each one is kept: name, then name#2, name#3, ..."""
    if kind != "agent":
        return name
    evals = {r["name"] for r in past if "name" in r and base_name(r["name"]) == name}
    return f"{name}#{len(evals) + 1}" if evals else name


def run_path(name: str) -> pathlib.Path:
    return HOME / "runs" / f"{name}.json.gz"


def read_run(name: str) -> dict:
    try:
        with gzip.open(run_path(name), "rt") as f:
            run = json.load(f)
    except FileNotFoundError:
        raise Fail(f"no saved run {name}; lab runs lists them") from None
    except (OSError, EOFError, ValueError):
        raise Fail(f"run {name} cannot be read: {run_path(name)} is damaged") from None
    if not isinstance(run, dict) or run.get("format") != RUN_FORMAT or not isinstance(run.get("sets"), dict):
        raise Fail(f"run {name} was saved by another version of lab; lab eval its file again")
    return run


def save_run(run: dict) -> None:
    """Writes run (a dataset evaluated again replaces its earlier scores; other datasets are kept) and removes all
    but the newest KEEP_RUNS runs."""
    path = run_path(run["name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(Fail):
        old = read_run(run["name"])
        if old.get("id") == run["id"]:
            run = {**run, "sets": {**old["sets"], **run["sets"]}}
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(run, f)
    os.replace(tmp, path)
    for stale in sorted(path.parent.glob("*.json.gz"), key=lambda p: p.stat().st_mtime, reverse=True)[KEEP_RUNS:]:
        stale.unlink(missing_ok=True)


def keep_run(path: pathlib.Path, kind: str, at: float, results: list[Result], questions: dict[str, list[dict]]) -> None:
    """Saves the eval as a run (questions: per dataset, each question's id, text, score and top documents), logs it in
    evals.jsonl and names it."""
    run_id = config_id(path)
    past = history()
    name = eval_name(run_name(run_id, taken_names(past)), kind, past)
    save_run(dict(format=RUN_FORMAT, id=run_id, name=name, at=at, file=str(path), kind=kind, data=lab.DATA_VERSION,
                  sets={r.dataset: dict(score=r.score, lo=r.lo, hi=r.hi, questions=questions[r.dataset]) for r in results}))
    for r in results:
        remember(path, kind, at, r, run_id, name)
    console.print()
    ok(f"saved as [me]{name}[/]")


def saved_runs(past: list[dict]) -> list[dict]:
    """The saved runs, newest first: name, file, kind, at (its latest eval) and score per dataset. Runs past the
    newest KEEP_RUNS, whose files are gone, are left out."""
    runs: dict[str, dict] = {}
    for r in sorted(past, key=lambda r: r["at"]):
        if isinstance(r.get("name"), str) and NAME.fullmatch(r["name"]):
            run = runs.setdefault(r["name"], dict(name=r["name"], scores={}))
            run.update(file=r["file"], kind=r["kind"], at=r["at"])
            run["scores"][r["dataset"]] = r["score"]
    return sorted((r for r in runs.values() if run_path(r["name"]).exists()), key=lambda r: -r["at"])


class QuietProgress(Progress):
    """Rich's Progress, minus the blank line it prints on stop when the output is not a terminal (setup.sh's tee)."""

    def stop(self) -> None:
        self.live.stop()


class AgentProgress(QuietProgress):
    """The progress bar with one line under it: the latest step any running agent took (now)."""

    now = Text("waiting for the model", "dim")

    def get_renderables(self):
        yield self.make_tasks_table(self.tasks)
        yield self.now


def latest_step(live: dict[int, list[dict]], seen: dict[int, int], short: str) -> Text | None:
    """The newest step among the running agents' traces (live: question index -> its turns, filling as it runs), as
    "crm 3  search  country=Finland …"; None when none took a step since the last look (seen: turns already shown)."""
    for i, turns in list(live.items()):
        if len(turns) > seen.get(i, 0):
            seen[i] = len(turns)
            return Text.assemble((f"{short} {i + 1}  ", "dim"), step_text(turns[-1]))
    return None


def step_text(turn: dict) -> Text:
    """A turn as one status line: its last tool call and arguments, else the text the model wrote."""
    call = turn["calls"][-1] if turn["calls"] else None
    return Text.assemble((call["name"], "acc"), "  ", args_text(call["args"])) if call else Text(turn["text"] or "answers")


def progress(*extra) -> Progress:
    return QuietProgress(SpinnerColumn(style="acc"), TextColumn("{task.description}"), BarColumn(complete_style="acc"), *extra,
                    TimeElapsedColumn(), console=console, transient=True)


# ---- the eval screen

@dataclass
class Result:
    dataset: str
    per: dict[str, float]  # question -> its score, rounded as the saved run keeps it
    score: float
    lo: float
    hi: float


def result(dataset: str, per: dict[str, float]) -> Result:
    x = np.array(list(per.values()), dtype=float)
    lo, hi = lab.bootstrap(x)
    return Result(dataset, {q: round(float(s), 4) for q, s in per.items()}, float(x.mean()), lo, hi)


def when(at: float) -> str:
    """The time of day in the room's time zone (lab connect saves it; a GPU box runs on UTC), with the weekday unless
    it is today."""
    tz = lab.room_tz()
    t = datetime.fromtimestamp(at, tz)
    return t.strftime("%H:%M" if t.date() == datetime.now(tz).date() else "%a %H:%M")


def result_table(path: pathlib.Path, metric: str, results: list[Result], elapsed: float, caption: str = "") -> Table:
    """This run's score on each dataset, with its 95% interval."""
    title = f"{file_key(str(path))} · {metric} · {elapsed:.1f} s"
    table = Table(title=title, title_justify="left", title_style="bold", box=None, padding=(0, 1), min_width=len(title),
                  caption=caption or None, caption_style="dim", caption_justify="left")
    for header, justify in (("dataset", "left"), (metric, "right"), ("", "left")):
        table.add_column(header, justify=justify, header_style="dim", no_wrap=True)
    for r in results:
        table.add_row(r.dataset, Text(f"{r.score:.3f}", "num"), Text(f"[{r.lo:.3f}, {r.hi:.3f}]", "dim"))
    return table


def show_end(table: Table) -> None:
    console.print()
    console.print(table)


def quiet_models() -> None:
    """No model-loading bars or Hugging Face notices among the eval's output; the CLI reports cache misses itself."""
    for name, value in (("HF_HUB_DISABLE_PROGRESS_BARS", "1"), ("TRANSFORMERS_VERBOSITY", "error"),
                        ("HF_HUB_VERBOSITY", "error"), ("TOKENIZERS_PARALLELISM", "false")):
        os.environ.setdefault(name, value)
    warnings.simplefilter("ignore", lab.RerankCacheMiss)


# ---- eval

NO_RERANKER = "the reranker is not part of the graded search (only of the notebook); take it out of search.py"


@contextlib.contextmanager
def without_reranker():
    lab.RERANKER_OFF = NO_RERANKER
    try:
        yield
    finally:
        lab.RERANKER_OFF = ""


def eval_search(path: pathlib.Path, module, worst: int, only: str | None) -> None:
    search = checked_search(module, path)
    run = time.time()
    with console.status("loading the data"):
        sets = [(f"{name} {split}", lab.load(name, split)) for name, split in EVAL_SETS if only in (None, name)]
    scores, ranked, tops = {}, {}, {}
    with progress(MofNCompleteColumn()) as bar:
        tasks = {label: bar.add_task(label, total=len(ds.qids)) for label, ds in sets}
        for label, ds in sets:
            for q in ds.qids:
                found = search(ds, ds.queries[q], k=TOP)
                rows = [r for r in found[:10] if ds.ids[r] not in ds.hidden[q]]
                ranked[label, q] = rows
                tops[label, q] = [ds.ids[r] for r in found if ds.ids[r] not in ds.hidden[q]]
                scores[label, q] = lab.ndcg10([ds.ids[i] for i in rows], ds.qrels[q])
                bar.advance(tasks[label])
    elapsed = time.time() - run
    results = [result(label, {q: scores[label, q] for q in ds.qids}) for label, ds in sets]
    if worst:
        show_worst(sets, scores, ranked, worst)
    show_end(result_table(path, "nDCG@10", results, elapsed))
    if lab.rerank_misses[0]:
        console.print(f"[warn]![/] {lab.rerank_misses[0]:,} reranker pairs missed the cache and scored lowest (no GPU to "
                      "score them)")
    by_label = {r.dataset: r for r in results}
    keep_run(path, "search", run, results, {label: [dict(id=q, text=ds.queries[q], score=by_label[label].per[q],
                                                         top=tops[label, q]) for q in ds.qids] for label, ds in sets})


def show_worst(sets, scores: dict, ranked: dict, n: int) -> None:
    """The n worst questions: your top 10 with their grades (2 and 1 relevant, . not), then relevant documents it missed."""
    by_label = dict(sets)
    for label, q in sorted(scores, key=scores.get)[:n]:
        ds = by_label[label]
        grades = ds.qrels[q]
        console.print(f"\n[num]{scores[label, q]:.3f}[/]  [dim]{DATASET_SHORT[label]} {ds.qids.index(q) + 1} · id {q}[/]")
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
    where = "on this machine" if url.startswith(lab.VLLM_URL) else "through the API"
    answer = checked_answer(module, path)
    questions = lab.crm_dev()
    found: dict[int, set[str]] = {}
    traces: dict[int, list[dict]] = {}
    run = time.time()
    live: dict[int, list[dict]] = {}  # each running question's turns, filled as its agent works

    def one(i: int, question: str) -> tuple[set[str], list[dict]]:
        with lab.tracing() as turns:
            live[i] = turns
            try:
                return answer(question), turns
            finally:
                live.pop(i, None)

    pool = ThreadPoolExecutor(lab.AGENT_THREADS)
    try:
        bar = AgentProgress(SpinnerColumn(style="acc"), TextColumn("{task.description}"), BarColumn(complete_style="acc"),
                            MofNCompleteColumn(), TimeElapsedColumn(), console=console, transient=True)
        with bar:
            task = bar.add_task(AGENT_SET, total=len(questions))
            futures = {pool.submit(one, i, q): i for i, (q, _) in enumerate(questions)}
            seen: dict[int, int] = {}
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                for f in done:
                    found[futures[f]], traces[futures[f]] = f.result()
                    bar.advance(task)
                step = latest_step(live, seen, DATASET_SHORT[AGENT_SET])
                if step is not None:
                    step.truncate(console.width - 1, overflow="ellipsis")
                    bar.now = step
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # an error or Ctrl-C does not wait for the running agents
    elapsed = time.time() - run
    res = result(AGENT_SET, {q: float(found[i] == gold) for i, (q, gold) in enumerate(questions)})
    short = DATASET_SHORT[AGENT_SET]
    for i in [i for i, (q, _) in enumerate(questions) if not res.per[q]][:worst]:
        q, gold = questions[i]
        console.print(Text.assemble("\n", ("✗ ", "err"), (f"{short} {i + 1}  ", "dim"), q))
        console.print(f"  [dim]expected[/] {', '.join(sorted(gold)) or 'none'}")
        console.print(f"  [dim]missing [/] {', '.join(sorted(gold - found[i])) or '–'}")
        console.print(f"  [dim]extra   [/] {', '.join(sorted(found[i] - gold)) or '–'}")
    console.print()
    width = len(f"{short} {len(questions)}")
    for i, (q, _) in enumerate(questions, 1):
        console.print(Text.assemble("  ", ("✓ ", "ok") if res.per[q] else ("✗ ", "err"), (f"{short} {i}".ljust(width), "dim"),
                                    "  ", q), no_wrap=True, overflow="ellipsis")
    show_end(result_table(path, "exact", [res], elapsed, f"agent model {model} {where}"))
    keep_run(path, "agent", run, [res], {AGENT_SET: [dict(id=q, text=q, score=res.per[q], top=sorted(found[i]),
                                                          expected=sorted(gold), trace=traces[i])
                                                     for i, (q, gold) in enumerate(questions)]})


def cmd_eval(args) -> None:
    quiet_models()
    path, module = load_file(named_file(args.file))
    kind = file_kind(module)
    if kind == "search":
        if (path.parent / "app" / "services.xml").is_file() and (path.parent / "feed.py").is_file():
            vespa_sync(path.parent)
        with without_reranker():
            eval_search(path, module, args.worst, args.data)
    elif args.data:
        raise Fail("--data picks a search dataset; an agent file runs on the CRM questions")
    else:
        eval_agent(path, module, args.worst)


# ---- runs, compare

DATASETS = {label.split()[0]: label for label in [*EVAL_LABELS, AGENT_SET]}  # fiqa -> fiqa dev
DATASET_SHORT = {label: short for short, label in DATASETS.items()}


def selected(runs: list[dict], kind: str, file: str | None) -> list[dict]:
    """runs of one file (a path), else of one kind (search, agent)."""
    if file is None:
        return [r for r in runs if r["kind"] == kind]
    key = file_key(str(pathlib.Path(file).resolve()))
    return [r for r in runs if file_key(r["file"]) == key]


def matching(runs: list[dict], word: str) -> list[dict]:
    """The runs (newest first) word names: NAME#N is one eval of an agent configuration (NAME#1 its first, saved as
    NAME); NAME, or the start of one, is the newest eval of each configuration it names."""
    if "#" in word:
        name, _, n = word.partition("#")
        return [r for r in runs if r["name"] == (name if n == "1" else word)]
    found = [r for r in runs if base_name(r["name"]) == word] or [r for r in runs if base_name(r["name"]).startswith(word)]
    newest: dict[str, dict] = {}
    for r in found:
        newest.setdefault(base_name(r["name"]), r)
    return list(newest.values())


def pick_run(runs: list[dict], word: str) -> dict:
    found = matching(runs, word)
    if not found:
        raise Fail(f"no run {word}; lab runs lists them")
    if len(found) > 1:
        raise Fail(f"{word} could be {', '.join(base_name(r['name']) for r in found[:5])}")
    return found[0]


def pair(runs: list[dict], picked: list[dict], kind: str, file: str | None) -> tuple[dict, dict]:
    """The two runs to compare, older first: the two picked; one picked and the newest other run of its file; else
    the last two runs of the file of the newest run of kind (or of file)."""
    if len(picked) > 2:
        raise Fail("lab compare takes at most two runs")
    if len(picked) == 2:
        if picked[0]["name"] == picked[1]["name"]:
            raise Fail(f"{picked[0]['name']} twice; name two runs")
        return picked[0], picked[1]
    if picked:
        key = file_key(picked[0]["file"])
        others = [r for r in runs if file_key(r["file"]) == key and r["name"] != picked[0]["name"]]
        if not others:
            raise Fail(f"{picked[0]['name']} is the only run of {key}")
        older, newer = sorted((picked[0], others[0]), key=lambda r: r["at"])
        return older, newer
    pool = selected(runs, kind, file)
    if not pool:
        raise Fail(f"no saved runs of {file or kind} yet; lab eval {file or kind} saves one")
    key = file_key(pool[0]["file"])
    same = [r for r in pool if file_key(r["file"]) == key]
    if len(same) < 2:
        raise Fail(f"{same[0]['name']} is the only run of {key} so far; change it and lab eval it again")
    return same[1], same[0]


def score_cell(value: float | None, top: float) -> Text:
    if value is None:
        return Text("–  ", "dim")
    return Text.assemble(f"{value:.3f}", (" ★", "ok") if value == top else "  ")


def runs_list(runs: list[dict], n: int) -> Table:
    """runs (newest first): the newest n, with each dataset's score (★ the best of all runs), the file and the time."""
    datasets = [d for d in [*EVAL_LABELS, AGENT_SET] if any(d in r["scores"] for r in runs)]
    top = {d: max(r["scores"][d] for r in runs if d in r["scores"]) for d in datasets}
    table = Table(box=None, padding=(0, 1), header_style="dim", caption_style="dim", caption_justify="left")
    table.add_column("run", style="me", no_wrap=True)
    for d in datasets:
        table.add_column(d, justify="right", no_wrap=True)
    table.add_column("file", overflow="fold")
    table.add_column("when", style="dim", no_wrap=True)
    for r in runs[:n]:
        table.add_row(r["name"], *(score_cell(r["scores"].get(d), top[d]) for d in datasets), file_key(r["file"]),
                      when(r["at"]))
    table.caption = "★ the best of these runs" + (f" · the newest {n} of {len(runs)} (-n for more)" if len(runs) > n else "")
    return table


def cmd_runs(args) -> None:
    what = args.what or "search"
    kind, file = (what, None) if what in ("search", "agent") else ("search", what)
    runs = selected(saved_runs(history()), kind, file)
    if not runs:
        raise Fail(f"no saved runs of {what} yet; lab eval {what} saves one")
    console.print(runs_list(runs, args.n))


def cmd_compare(args) -> None:
    words = list(args.words)
    kind = words.pop(0) if words and words[0] in ("search", "agent") else None
    if words and words[0] in ("question", "q"):
        compare_question(words[1:], kind)
        return
    file = next((w for w in words if w.endswith(".py")), None)
    runs = saved_runs(history())
    a, b = pair(runs, [pick_run(runs, w) for w in words if w != file], kind or "search", file)
    show_compare(read_run(a["name"]), read_run(b["name"]), args.n)


def files_of(a: dict, b: dict) -> str:
    ka, kb = file_key(a["file"]), file_key(b["file"])
    return ka if ka == kb else f"{ka} → {kb}"


def run_line(a: dict, b: dict) -> Text:
    return Text.assemble((a["name"], "me"), (f" {when(a['at'])}", "dim"), " → ", (b["name"], "me"),
                         (f" {when(b['at'])}", "dim"), (f" · {files_of(a, b)}", "dim"))


def change_style(delta: float) -> str:
    return "ok" if delta > 0 else "err" if delta < 0 else "dim"


def cut(text: str, n: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n - 1] + "…"


def changes(qa: list[dict], qb: list[dict]) -> list[tuple[int, dict, dict]]:
    """The questions in both runs, numbered as in the data: (number, the question in a, in b)."""
    before = {q["id"]: q for q in qa}
    return [(i, before[q["id"]], q) for i, q in enumerate(qb, 1) if q["id"] in before]


def moved_view(label: str, sa: dict, sb: dict, n: int) -> Group:
    """One dataset of a comparison: its score a → b, how many questions went up, down or stayed, and the n that
    moved most each way."""
    rows = [(i, a, b, a["score"], b["score"]) for i, a, b in changes(sa["questions"], sb["questions"])]
    up = sorted((r for r in rows if r[4] > r[3]), key=lambda r: (r[3] - r[4], r[0]))
    down = sorted((r for r in rows if r[4] < r[3]), key=lambda r: (r[4] - r[3], r[0]))
    delta = sb["score"] - sa["score"]
    title = Text.assemble((f"{label:<10}", "bold"), f"{sa['score']:.3f} → ", (f"{sb['score']:.3f}", "num"), "  ",
                          (f"{delta:+.3f}", change_style(delta)), "   ", (f"↑{len(up)}", "ok"), " ", (f"↓{len(down)}", "err"),
                          " ", (f"={len(rows) - len(up) - len(down)}", "dim"))
    if not up and not down:
        return Group(title, Text("  no question moved", "dim"))
    table = Table(box=None, padding=(0, 1), show_header=False, expand=True)
    table.add_column(no_wrap=True, min_width=1)
    table.add_column(no_wrap=True, min_width=8)
    table.add_column(no_wrap=True, min_width=13)
    table.add_column(justify="right", no_wrap=True, min_width=6)
    shown = up[:n] + down[:n]
    traced_both = all("trace" in qa and "trace" in qb for _, qa, qb, _, _ in shown)
    if traced_both:
        table.add_column(no_wrap=True)
    table.add_column(no_wrap=True, overflow="ellipsis", ratio=1)
    for i, qa, qb, x, y in shown:
        table.add_row("", f"{DATASET_SHORT[label]} {i}", f"{x:.3f} → {y:.3f}", Text(f"{y - x:+.3f}", change_style(y - x)),
                      *([Text(effort_change(qa["trace"], qb["trace"]), "dim")] if traced_both else []), cut(qb["text"]))
    return Group(title, table)


def show_compare(a: dict, b: dict, n: int) -> None:
    labels = [label for label in b["sets"] if label in a["sets"]]
    if not labels:
        raise Fail(f"{a['name']} and {b['name']} share no dataset")
    console.print(run_line(a, b))
    for label in labels:
        console.print()
        console.print(moved_view(label, a["sets"][label], b["sets"][label], n))


def compare_question(words: list[str], kind: str | None) -> None:
    """lab compare [search|agent] question [DATASET] NUMBER-OR-WORDS [RUN RUN]"""
    runs = saved_runs(history())
    file = next((w for w in words if w.endswith(".py")), None)
    words = [w for w in words if w != file]
    picked: list[dict] = []
    for _ in range(2):  # run names come last; they all have a hyphen, question words seldom do
        found = matching(runs, words[-1]) if words and "-" in words[-1] else []
        if len(found) == 1:
            picked.insert(0, found[0])
            words.pop()
    dataset = DATASETS.get(words[0]) if words else None
    if dataset:
        words.pop(0)
        kind = kind or ("agent" if dataset == AGENT_SET else "search")
    query = " ".join(words).strip()
    if not query:
        raise Fail("which question? lab compare question NUMBER, or a few words of it")
    a, b = pair(runs, picked, kind or "search", file)
    ra, rb = read_run(a["name"]), read_run(b["name"])
    label, n, qa, qb = find_question(ra, rb, dataset, query)
    (drill_agent if label == AGENT_SET else drill_search)(ra, rb, label, n, qa, qb)


def find_question(a: dict, b: dict, dataset: str | None, query: str) -> tuple[str, int, dict, dict]:
    """The question query names in both runs: its number in the data, its id, or words of its text."""
    labels = [label for label in b["sets"] if label in a["sets"] and dataset in (None, label)]
    if not labels:
        raise Fail(f"{a['name']} and {b['name']} share no {dataset or 'dataset'}")
    found = []
    for label in labels:
        for i, qa, qb in changes(a["sets"][label]["questions"], b["sets"][label]["questions"]):
            if str(i) == query if query.isdigit() else query == qb["id"] or query.lower() in qb["text"].lower():
                found.append((label, i, qa, qb))
    if len(found) == 1:
        return found[0]
    if not found:
        raise Fail(f"no question {query} in {' or '.join(labels)}")
    for label, i, _, q in found[:10]:
        console.print(Text.assemble(f"  {DATASET_SHORT[label]} {i:<4} ", (cut(q["text"]), "dim")), no_wrap=True,
                      overflow="ellipsis")
    more = f" (the first 10 of {len(found)})" if len(found) > 10 else ""
    raise Fail(f"{len(found)} questions match {query}{more}; name one, as in lab compare question "
               f"{DATASET_SHORT[found[0][0]]} {found[0][1]}")


def question_head(a: dict, b: dict, label: str, n: int, q: dict, metric: str, x: float, y: float) -> None:
    console.print(Text.assemble((f"{DATASET_SHORT[label]} {n}", "bold"), "  ", run_line(a, b)))
    console.print(Text("  " + " ".join(q["text"].split())))
    console.print(Text.assemble(f"  {metric}  {x:.3f} → ", (f"{y:.3f}", "num"), "  ", (f"{y - x:+.3f}", change_style(y - x))))


def grade_mark(grade: int) -> tuple[str, str]:
    return {2: ("✓", "grade2"), 1: ("✓", "ok")}.get(grade, ("·", "dim"))


def doc_text(ds: lab.Dataset, doc: str) -> str:
    i = ds.row[doc]
    return " ".join(f"{ds.titles[i]} {ds.bodies[i]}".split())[:300]


def rank_change(was: int | None, now: int | None) -> Text:
    """"7 → 2 ↑": a document's rank in a and in b; – is not in the top TOP."""
    arrow = ("↑", "ok") if now and (not was or now < was) else ("↓", "err") if was and (not now or now > was) else ""
    return Text.assemble(f"{was or '–':>2} → {now or '–':<2} ", arrow)


def drill_search(a: dict, b: dict, label: str, n: int, qa: dict, qb: dict) -> None:
    """One question in two runs: its relevant documents' ranks a → b (those in either top TOP), then both top 10s side
    by side, the newer run (b) first."""
    name, split = label.split()
    with console.status("loading the data"):
        ds = lab.load(name, split)
    hidden = ds.hidden.get(qb["id"], set())
    grades = {d: g for d, g in ds.qrels.get(qb["id"], {}).items() if g > 0 and d in ds.row and d not in hidden}
    in_a = {d: i for i, d in enumerate(qa["top"], 1)}
    in_b = {d: i for i, d in enumerate(qb["top"], 1)}
    question_head(a, b, label, n, qb, "nDCG@10", qa["score"], qb["score"])

    relevant = sorted(grades, key=lambda d: (-grades[d], min(in_a.get(d, TOP + 1), in_b.get(d, TOP + 1))))
    found = [d for d in relevant if d in in_a or d in in_b]
    notes = [f"– not in the top {TOP}"] if any(d not in in_a or d not in in_b for d in found[:10]) else []
    notes += [f"{len(found) - 10} more relevant"] if len(found) > 10 else []
    notes += [f"{len(relevant) - len(found)} relevant in neither top {TOP}"] if len(found) < len(relevant) else []
    table = Table(box=None, padding=(0, 1), header_style="dim", caption_style="dim", caption_justify="left", expand=True,
                  caption=" · ".join(notes) or None)
    table.add_column("", no_wrap=True, min_width=1)
    table.add_column("relevant", no_wrap=True, overflow="ellipsis", ratio=1)
    table.add_column("rank", no_wrap=True, min_width=9)
    for d in found[:10]:
        table.add_row(Text(*grade_mark(grades[d])), doc_text(ds, d), rank_change(in_a.get(d), in_b.get(d)))
    console.print()
    console.print(table if found else Text(" · ".join(notes), "dim"))

    side = Table(box=None, padding=(0, 1), header_style="dim", caption_style="dim", caption_justify="left", expand=True)
    for header, width in (("", 2), ("", 1), (b["name"], None), ("", 3), ("", 1), ("", 2), ("", 1), (a["name"], None),
                          ("", 3)):
        side.add_column(header, no_wrap=True, overflow="ellipsis", ratio=None if width else 1, min_width=width,
                        justify="left" if width in (None, 1) else "right")
    for i in range(10):
        side.add_row(*top_cells(ds, grades, qb["top"], in_a, i, "new"), Text("│", "dim"),
                     *top_cells(ds, grades, qa["top"], in_b, i, "out"))
    side.caption = (f"beside each document: its rank in the other run · new: not in {a['name']}'s top {TOP} · out: "
                    f"not in {b['name']}'s")
    console.print()
    console.print(side)


def top_cells(ds: lab.Dataset, grades: dict[str, int], top: list[str], other: dict[str, int], i: int,
              missing: str) -> list:
    if i >= len(top):
        return ["", "", "", ""]
    d, grade = top[i], grades.get(top[i], 0)
    return [str(i + 1), Text(*grade_mark(grade)), Text(doc_text(ds, d) if d in ds.row else d, "" if grade else "dim"),
            Text(str(other[d]) if d in other else missing, "" if d in other else "dim")]


def drill_agent(a: dict, b: dict, label: str, n: int, qa: dict, qb: dict) -> None:
    """One agent question in two runs: the deals expected, and what each run found."""
    question_head(a, b, label, n, qb, "exact", qa["score"], qb["score"])
    expected = set(qb.get("expected", []))
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column()
    grid.add_row("  expected", ", ".join(sorted(expected)) or "none")
    for run, q in ((a, qa), (b, qb)):
        got = set(q["top"])
        grid.add_row(f"  {run['name']}", Text.assemble(("✓ ", "ok") if got == expected else ("✗ ", "err"),
                                                       f"found {', '.join(sorted(got)) or 'none'}",
                                                       (f" · missing {', '.join(sorted(expected - got)) or '–'}"
                                                        f" · extra {', '.join(sorted(got - expected)) or '–'}", "dim")))
    console.print(grid)
    mark = first_difference(qa, qb)
    for run, q in ((a, qa), (b, qb)):
        console.print()
        if "trace" not in q:
            console.print(f"  [me]{run['name']}[/] [dim]was saved before lab eval recorded trajectories[/]")
            continue
        console.print(Text.assemble("  ", (run["name"], "me"), " · ", verdict(q), (f" · {trace_totals(q['trace'])}", "dim")))
        for line in trace_lines(q, mark, console.width, outcome=False):
            console.print(line, no_wrap=True, overflow="ellipsis")


# ---- trace: one agent question's trajectory, recorded by lab eval agent

def trace_totals(trace: list[dict]) -> str:
    """"3 turns · 6,210 tokens · $0.004"; tokens and cost when the model reported them."""
    tokens = [t["tokens"] for t in trace if t["tokens"] is not None]
    usd = [t["usd"] for t in trace if t["usd"] is not None]
    return " · ".join([f"{len(trace)} turn{'' if len(trace) == 1 else 's'}", *([f"{sum(tokens):,} tokens"] if tokens else []),
                       *([f"${sum(usd):.3f}"] if usd else [])])


def short_tokens(trace: list[dict]) -> str:
    n = sum(t["tokens"] or 0 for t in trace)
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def effort_change(ta: list[dict], tb: list[dict]) -> str:
    """"3→5 turns 6.2k→8.1k tok": a question's turns and tokens in run a → b."""
    return f"{len(ta)}→{len(tb)} turns {short_tokens(ta)}→{short_tokens(tb)} tok"


def verdict(q: dict) -> Text:
    """"✗ found 4 of 6, 1 extra"."""
    got, expected = set(q["top"]), set(q["expected"])
    extra = len(got - expected)
    return Text.assemble(("✓ ", "ok") if got == expected else ("✗ ", "err"),
                         f"found {len(got & expected)} of {len(expected)}" + (f", {extra} extra" if extra else ""))


def value_text(v, top: bool = False) -> str:
    """An argument as the model sent it, without Python's quotes: text="pharma", {industry: Pharmaceuticals}."""
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {value_text(x)}" for k, x in v.items()) + "}"
    if isinstance(v, list):
        return "[" + ", ".join(value_text(x) for x in v) + "]"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False) if top else v
    return json.dumps(v)


def args_text(args: dict | str) -> str:
    """name=value pairs, leaving out the arguments the model sent as null; arguments that are not a JSON object as sent."""
    if isinstance(args, str):
        return args
    return " ".join(f"{k}={value_text(v, top=True)}" for k, v in args.items() if v is not None)


def steps_of(trace: list[dict]) -> list[tuple[str, str]]:
    """The tool calls in order, as (tool, arguments)."""
    return [(c["name"], args_text(c["args"])) for t in trace for c in t["calls"]]


def first_difference(qa: dict, qb: dict) -> int | None:
    """Where two runs of a question part: the first tool call that differs, else len(calls) when only their answers
    differ, else None."""
    if "trace" not in qa or "trace" not in qb:
        return None
    sa, sb = steps_of(qa["trace"]), steps_of(qb["trace"])
    for k, (x, y) in enumerate(zip(sa, sb)):
        if x != y:
            return k
    if len(sa) != len(sb):
        return min(len(sa), len(sb))
    return None if qa["top"] == qb["top"] else len(sa)


def trace_lines(q: dict, mark: int | None, width: int, outcome: bool = True) -> list[Text]:
    """The trajectory, one line per tool call: turn, tool, arguments → what it returned, and the text the model wrote;
    then its answer and, with outcome, the deals expected, missing and extra. mark: the tool call (counted over the
    whole trace; len(calls) is the answer) that a › marks."""
    trace, got, expected = q["trace"], sorted(q["top"]), sorted(q["expected"])
    num_w = len(str(len(trace)))
    name_w = max([len("answer"), *(len(c["name"]) for t in trace for c in t["calls"])])
    lead = 2 + num_w + 2 + name_w + 2
    rest = max(24, width - lead)
    arg_w = min(max([0, *(len(a) for _, a in steps_of(trace))]), (rest - 4) * 2 // 3)
    lines: list[Text] = []

    def add(turn: str, name: str, body: Text, marked: bool = False) -> None:
        lines.append(Text.assemble(("›" if marked else " ", "warn"), " ", f"{turn:>{num_w}}  {name:<{name_w}}  ", body))

    k = 0
    if not trace:
        lines.append(Text("  no lab.chat calls recorded", "dim"))
    for n, t in enumerate(trace, 1):
        turn = str(n)
        for c in t["calls"]:
            result = c["result"]
            add(turn, c["name"], Text.assemble(cut(args_text(c["args"]), arg_w).ljust(arg_w), "  → ",
                                               (cut(result, rest - arg_w - 4), "") if result else ("–", "dim")), k == mark)
            turn, k = "", k + 1
        if n == len(trace) and not t["calls"]:
            add(turn, "answer", Text(" ".join(got) or "none"), k == mark)
            turn = ""
        if t["text"]:
            add(turn, "says", Text(cut(t["text"], rest), "dim"))
    if not trace or trace[-1]["calls"]:
        add("", "answer", Text(" ".join(got) or "none"), k == mark)
    if outcome:
        pad = " " * (lead - 2 - len("expected"))
        lines.append(Text.assemble(("  expected", "dim"), pad, " ".join(expected) or "none"))
        missing, extra = sorted(set(expected) - set(got)), sorted(set(got) - set(expected))
        if missing or extra:
            lines.append(Text.assemble(("  missing ", "dim"), pad, " ".join(missing) or "–", ("      extra  ", "dim"),
                                       " ".join(extra) or "–"))
    return lines


def cmd_trace(args) -> None:
    """lab trace [crm] NUMBER-OR-WORDS [RUN]: one question of an agent run (default: the newest), step by step."""
    runs = selected(saved_runs(history()), "agent", None)
    words = list(args.words)
    if words and DATASETS.get(words[0]) == AGENT_SET:
        words.pop(0)
    run = pick_run(runs, words.pop()) if len(words) > 1 and matching(runs, words[-1]) else runs[0] if runs else None
    if run is None:
        raise Fail("no saved runs of agent yet; lab eval agent saves one")
    query = " ".join(words).strip()
    if not query:
        raise Fail("which question? lab trace crm NUMBER, or a few words of it")
    saved = read_run(run["name"])
    label, n, q, _ = find_question(saved, saved, AGENT_SET, query)
    if "trace" not in q:
        raise Fail(f"{run['name']} was saved before lab eval recorded trajectories; lab eval its file again")
    if args.json:
        print(json.dumps(dict(run=run["name"], question=n, **q), ensure_ascii=False, indent=1))
        return
    console.print(Text.assemble((f"{DATASET_SHORT[label]} {n}", "bold"), "  ", (run["name"], "me"), " · ", verdict(q),
                                (f" · {trace_totals(q['trace'])}", "dim")))
    console.print(Text("  " + " ".join(q["text"].split())))
    for line in trace_lines(q, None, console.width):
        console.print(line, no_wrap=True, overflow="ellipsis")


# ---- query

def cmd_query(args) -> None:
    """lab query search|agent TEXT: your search's top k for any question, or your agent's answer with its trace."""
    quiet_models()
    question = " ".join(args.text).strip()
    if not question:
        raise Fail(f"what to ask? lab query {args.what} \"your question\"")
    path, module = load_file(named_file(args.what))
    if file_kind(module) != args.what:
        raise Fail(f"{path.name} is not a {args.what} file")
    if args.what == "search":
        with without_reranker():
            query_search(path, module, question, args.data or "fiqa", args.k)
    elif args.data:
        raise Fail("--data picks a search dataset; the agent asks the CRM")
    else:
        query_agent(path, module, question)


def query_search(path: pathlib.Path, module, question: str, name: str, k: int) -> None:
    if (path.parent / "app" / "services.xml").is_file() and (path.parent / "feed.py").is_file():
        vespa_sync(path.parent)
    with console.status("loading the data"):
        ds = lab.load(name, dict(EVAL_SETS)[name])
    start = time.time()
    with console.status("searching (a new question loads the question encoder: a few seconds)"):
        rows = checked_search(module, path)(ds, question, k=min(k, len(ds.ids)))
    stems = {lab.stem(w) for w in re.findall(r"\w+", lab.normalize(question)) if w not in lab.STOPWORDS}
    table = Table(box=None, padding=(0, 1), header_style="dim", caption_style="dim", caption_justify="left",
                  width=min(console.width, QUERY_WIDTH),
                  caption=f"{name} · top {len(rows)} of {len(ds.ids):,} documents · {time.time() - start:.1f} s")
    table.add_column("", style="dim", justify="right", no_wrap=True)
    table.add_column("document", no_wrap=True, overflow="ellipsis", ratio=1)
    table.add_column("id", style="dim", no_wrap=True)
    for n, i in enumerate(rows, 1):
        text = Text.assemble(*([(ds.titles[i], "bold"), "  "] if ds.titles[i] else []))
        text.append_text(snippet(html.unescape(ds.bodies[i]), stems, QUERY_WIDTH // 2))
        table.add_row(str(n), text, ds.ids[i])
    console.print(table)


QUERY_WIDTH = 110  # lab query's table: a line of text, not the terminal's width


def snippet(text: str, stems: set[str], width: int) -> Text:
    """text on one line from a little before its first word with a stem in stems, those words highlighted."""
    words = list(re.finditer(r"\w+", text := " ".join(text.split())))
    hits = [m for m in words if lab.stem(lab.normalize(m.group())) in stems]
    start = text.rfind(" ", 0, hits[0].start() - width // 4) + 1 if hits and hits[0].start() > width // 2 else 0
    out = Text("…" if start else "")
    at = start
    for m in hits:
        if m.start() < start:
            continue
        out.append(text[at:m.start()])
        out.append(m.group(), "acc")
        at = m.end()
    out.append(text[at:])
    return out


DEAL_FIELDS = ("company", "stage", "country")


def query_agent(path: pathlib.Path, module, question: str) -> None:
    start = time.time()
    answer, live, result = checked_answer(module, path), [], {}

    def run() -> None:
        with lab.tracing() as turns:
            live.append(turns)
            try:
                result["found"] = answer(question)
            except BaseException as e:  # re-raised below, in the CLI's thread
                result["error"] = e

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    with console.status("asking the model") as status:
        shown = 0
        while worker.is_alive():
            worker.join(0.25)
            turns = live[0] if live else []
            if len(turns) > shown:
                shown = len(turns)
                status.update(Text.assemble((f"turn {shown}  ", "dim"), step_text(turns[-1])))
    if "error" in result:
        raise result["error"]
    found, trace = result["found"], live[0]
    console.print(Text.assemble((file_key(str(path)), "bold"), f" · {trace_totals(trace)} · {time.time() - start:.1f} s"))
    q = dict(trace=trace, top=sorted(found), expected=[])
    for line in trace_lines(q, None, console.width, outcome=False):
        console.print(line, no_wrap=True, overflow="ellipsis")
    deals = {d["deal_id"]: d for d in lab.crm()}
    table = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    for d in sorted(found):
        deal = deals.get(d)
        if deal is None:
            table.add_row(d, Text("no such deal", "err"))
            continue
        table.add_row(d, *(deal[f] for f in DEAL_FIELDS), f"€{float(deal['amount_eur']):,.0f}",
                      f"closes {deal['close_date']}")
    if found:
        console.print()
        console.print(table)


# ---- submit

def cmd_submit(args) -> None:
    """Uploads the code the final board grades (search/, or the agent file) and sends the dev score of its latest saved
    lab eval to the live board."""
    label = None if args.m is None else args.m.strip()
    if label is not None and not 1 <= len(label) <= LABEL_CHARS:
        raise Fail(f"-m: a label of 1 to {LABEL_CHARS} characters on what your file does (yours has {len(label)})")
    if not lab.LAB.exists():
        raise Fail("not connected to the lab; lab connect URL TOKEN (both on your lab page)")
    word = args.file or "search"
    if word in ("search", "agent"):
        kind, path = word, pathlib.Path(named_file(word)).resolve()
    else:
        quiet_models()
        path, module = load_file(named_file(word))
        kind = file_kind(module)
    if kind == "agent":
        submit_agent_file(path)
    elif path == (pathlib.Path.cwd() / labvespa.FOLDER / "search.py").resolve():
        submit_search(path)
    else:
        warn(f"only search/ is graded for the final board, so {path.name} goes to the live board only; lab submit "
             "search uploads search/")
    post_dev(kind, path, label)


BOARD_SETS = {"search": EVAL_LABELS[0], "agent": AGENT_SET}  # the dev score each live board shows


def latest_scores(past: list[dict], kind: str, path: pathlib.Path) -> dict[str, float]:
    """Each dataset's score in the latest saved eval of path as it is now (its config_id): dataset -> score."""
    run_id = config_id(path)
    scores: dict[str, tuple[float, float]] = {}
    for r in past:
        if r["kind"] == kind and r.get("id") == run_id and r["at"] >= scores.get(r["dataset"], (-1.0, 0.0))[0]:
            scores[r["dataset"]] = (r["at"], r["score"])
    return {dataset: score for dataset, (_, score) in scores.items()}


def post_dev(kind: str, path: pathlib.Path, label: str | None) -> None:
    """Sends the latest saved eval's scores of path to the live board, or says to run lab eval first."""
    past = history()
    scores = latest_scores(past, kind, path)
    board_set = BOARD_SETS[kind]
    if board_set not in scores:
        warn(f"no saved {board_set} eval of {path.name} as it is now, so the live board keeps your last score; run "
             f"lab eval {'agent' if kind == 'agent' else 'search'}, then lab submit again")
        return
    label = label or run_name(config_id(path), taken_names(past))
    result = lab.call("/api/board" if kind == "search" else "/api/agent-board", dict(label=label, scores=scores))
    ok(f"live board: {board_set} [num]{result['score']:.4f}[/] ({label}) · rank [num]{result['rank']}[/] of {result['of']}")


# ---- submitting search/

HANDIN_BYTES = 2 * 1024 * 1024  # the lab's limit on a hand-in, compressed
HANDIN_FILE_BYTES = 8 * 1024 * 1024  # a larger file stays out of the hand-in


def pack(folder: pathlib.Path) -> tuple[bytes, list[str], list[str]]:
    """A gzip tar of folder's files, named relative to it; the files packed, and those left out (too large, links).
    Dotfiles, __pycache__ and .pyc never go in. The same files pack to the same bytes, so the lab sees an unchanged
    hand-in as unchanged."""
    packed, left_out = [], []
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for path in sorted(folder.rglob("*")):
            name = path.relative_to(folder).as_posix()
            if any(part.startswith(".") or part == "__pycache__" for part in name.split("/")) or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                left_out.append(f"{name} (a link)")
                continue
            if not path.is_file():
                continue
            size = path.stat().st_size
            if size > HANDIN_FILE_BYTES:
                left_out.append(f"{name} ({size / 1e6:.1f} MB)")
                continue
            info = tarfile.TarInfo(name)
            info.size, info.mode = size, 0o644
            with open(path, "rb") as f:
                tar.addfile(info, f)
            packed.append(name)
    return buf.getvalue(), packed, left_out


def send_handin(data: bytes, route: str) -> dict:
    """POSTs an upload (the packed search/ folder, or an agent file to /api/agent-handin) to the lab with the connected
    token; the lab's receipt."""
    conn = json.loads(lab.LAB.read_text())
    req = urllib.request.Request(conn["url"] + route, data,
                                 {"Content-Type": "application/octet-stream", "Authorization": f"Bearer {conn['token']}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise lab.LabError(e.code, lab.error_detail(e.read().decode(errors="replace")) or str(e.reason)) from None


SUBMITTED = ("search.py", "feed.py", "analyze.py", "app")  # what a search folder must hold; synonyms.json is optional


def require_eval(kind: str, path: pathlib.Path) -> None:
    """Fails unless path, as it is now, has a saved lab eval: an upload that does not run would only fail in grading."""
    board_set = BOARD_SETS[kind]
    if board_set not in latest_scores(history(), kind, path):
        raise Fail(f"no saved {board_set} eval of {path.name} as it is now; lab eval {kind} first (it catches errors "
                   f"before grading does), then lab submit {kind}")


def submit_search(path: pathlib.Path) -> None:
    name = labvespa.FOLDER.as_posix()
    folder = pathlib.Path.cwd() / name
    missing = [f for f in SUBMITTED if not (folder / f).exists()]
    if missing:
        raise Fail(f"no {', '.join(f'{name}/{f}' for f in missing)} in {pathlib.Path.cwd()}; {elsewhere(labvespa.FOLDER)}")
    data, packed, left_out = pack(folder)
    for f in left_out:
        warn(f"left out {name}/{f}")
    if len(data) > HANDIN_BYTES:
        raise Fail(f"{name}/ packs to {len(data) / 1e6:.1f} MB; the lab takes {HANDIN_BYTES / 1e6:.1f} MB. "
                   f"Move data and large models out of {name}/")
    require_eval("search", path)
    with console.status(f"submitting {name}/ ({len(packed)} files)"):
        got = send_handin(data, "/api/handin")
    ok(f"submitted {name}/ at {when(got['at'])} · {got['files']} files · {got['bytes'] / 1e3:.0f} kB · {GRADED_SOON}")


AGENT_BYTES = 256 * 1024  # the lab's limit on an agent file
GRADED_SOON = "graded in a few minutes · lab status shows it"


def submit_agent_file(path: pathlib.Path) -> None:
    """Uploads the agent file; the final round runs it on the hidden CRM questions."""
    data = path.read_bytes()
    if len(data) > AGENT_BYTES:
        raise Fail(f"{path.name} is {len(data) / 1e3:.0f} kB; the lab takes {AGENT_BYTES // 1024} kB")
    require_eval("agent", path)
    with console.status(f"submitting {path.name}"):
        got = send_handin(data, "/api/agent-handin")
    ok(f"submitted {path.name} at {when(got['at'])} · {GRADED_SOON}")


# ---- setup, serve, status

def gpu() -> tuple[str, int, int] | None:
    """(name, free bytes, total bytes) of GPU 0, or None without one."""
    import torch

    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info(0)
    return torch.cuda.get_device_name(0), free, total


def big_gpu(found: tuple[str, int, int] | None) -> bool:
    """Whether the GPU from gpu() can hold the agent model next to the reranker."""
    return found is not None and found[2] >= BIG_GPU_GB * 1e9


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


NO_GPU = "no GPU: the reranker from the precomputed cache, the agent through the API"


def cmd_setup(args) -> None:
    with console.status("checking the GPU"):
        found = gpu()
    if found is None:
        console.print(f"[warn]![/] {NO_GPU}")
    else:
        name, free, total = found
        # A running lab serve holds most of the GPU: setup run again leaves it be.
        if big_gpu(found) and free < BIG_GPU_GB * 1e9 and not args.force and lab.local_agent_model() is None:
            raise Fail(f"{name} has {gb(free)} free of {gb(total)}; other programs use the rest. Stop them, or pass "
                       "--force to try anyway")
        ok(f"GPU  {name} · {gb(free)} free of {gb(total)}")
        if not big_gpu(found):
            console.print(f"[warn]![/] {lab.AGENT_MODEL} needs a {BIG_GPU_GB} GB GPU; the agent runs through the API, "
                          "the reranker here")

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
        models += [("reranker", lab.RERANK_MODEL, None)]
    if big_gpu(found):
        models += [("agent model", lab.AGENT_MODEL, None)]
    for what, repo, rev in models:
        console.print(f"[acc]→[/] {what}  {repo}")
        snapshot_download(repo, revision=rev)
        ok(f"{what}  {repo}")

    if not big_gpu(found):
        return
    if vllm_installed():
        ok(f"vLLM {lab.VLLM_VERSION}")
    else:
        install_vllm()
        ok(f"vLLM {lab.VLLM_VERSION}  {VENV}")


def download(what: str, path: pathlib.Path, fetch) -> pathlib.Path | None:
    """fetch(progress) with a progress bar, unless path is there already."""
    if not path.exists():
        with QuietProgress(TextColumn(f"  {what}"), BarColumn(complete_style="acc"), DownloadColumn(), TransferSpeedColumn(),
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
    if total < BIG_GPU_GB * 1e9:
        raise Fail(f"the GPU has {gb(total)}; {lab.AGENT_MODEL} and the reranker need a card of {BIG_GPU_GB} GB or more")
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
        found = gpu()
        if not big_gpu(found):
            why = f"{found[0]} has {gb(found[2])}" if found else "no GPU here"
            ok(f"nothing to serve: {why}, so the agent runs through the API")
            return
        if not binary.exists():
            raise Fail("vLLM is not installed; run lab setup first (on a GPU machine; without one, chat goes "
                       "through the API and there is nothing to serve)")
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
                if not vllm_alive(proc):
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


def vllm_alive(proc: subprocess.Popen | None) -> bool:
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


HANDINS = (("search", "/api/handin", lambda x: f"{x:.3f}"), ("agent", "/api/agent-handin", lambda x: f"{round(x * 100)}%"))


def cmd_status(args) -> None:
    nick, me, handins = None, {}, []
    with console.status("checking"):
        found = gpu()
        model = lab.local_agent_model()
        try:
            me = lab.call("/api/me") if lab.LAB.exists() else {}
            nick = me.get("nick")
            if nick:
                lab.save_tz(me.get("tz"))
            conn = f"[ok]connected[/] as [num]{nick}[/] · {json.loads(lab.LAB.read_text())['url']}" if nick else \
                "[warn]not connected[/] · lab connect URL TOKEN"
            handins = [(kind, lab.call(route)["handin"], show) for kind, route, show in HANDINS] if nick else []
        except lab.LabError as e:
            conn = f"[err]refused[/] · {e}"
        except OSError as e:
            conn = f"[err]unreachable[/] · {getattr(e, 'reason', e)}"
        needed = [("encoder", lab.QUERY_MODEL, lab.QUERY_REVISION)] + (
            [("reranker", lab.RERANK_MODEL, None)] if found else []) + (
            [("agent", lab.AGENT_MODEL, None)] if big_gpu(found) else [])
        have = [lab.downloaded(repo, rev) for _, repo, rev in needed]
        models = " · ".join(f"{what} {'[ok]✓[/]' if got else '[err]✗[/]'}" for (what, _, _), got in zip(needed, have))
        scores = (lab.DATA / lab.DATA_VERSION / lab.rerank_cache_name()).exists()
        data = (lab.DATA / lab.DATA_VERSION).exists()
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column()
    grid.add_row("mode", "GPU" if found else f"[warn]{NO_GPU.removeprefix('no GPU: ')}[/]")
    grid.add_row("GPU", f"{found[0]} · {gb(found[1])} free of {gb(found[2])}" if found else "[warn]none[/]")
    grid.add_row("agent", f"[ok]{model}[/] on this machine" if model == lab.AGENT_MODEL else
                 f"{lab.LAB_AGENT_MODEL} through the API{budget_left(me)}"
                 + (f" · lab serve runs {lab.AGENT_MODEL} here" if big_gpu(found) else ""))
    grid.add_row("lab", conn)
    for kind, handin, show in handins:
        if handin is not None:
            line = Text.from_markup(handin_line(handin, show), overflow="ellipsis")
            line.no_wrap = True  # a failure's reason in full, cut at the terminal's width
            grid.add_row(f"{kind} upload", line)
    grid.add_row("data", f"{lab.DATA_VERSION} [ok]✓[/]" if data else "[err]missing[/] · lab setup")
    grid.add_row("models", models)
    grid.add_row("reranker scores", f"{lab.rerank_cache_name()} " + ("[ok]✓[/]" if scores else "[err]✗[/] · lab setup"))
    console.print(Panel(grid, title="lab status", title_align="left", border_style="acc", expand=False))


def handin_line(h: dict, show) -> str:
    """"10:07 · graded · hidden preview 0.598": the grading of the latest upload, from the lab's /api/handin or
    /api/agent-handin; show formats a score."""
    at = when(h["at"])
    if h["state"] == "graded":
        preview = "" if h["preview"] is None else f" · hidden preview [num]{show(h['preview'])}[/]"
        return f"{at} · [ok]graded[/]{preview}" + ("" if h["score"] is None else f" · hidden [num]{show(h['score'])}[/]")
    if h["state"] == "failed":
        return f"{at} · [err]failed[/]: {escape(h['error'] or '')}"
    if h["state"] == "queued" and h.get("position"):
        return f"{at} · queued, {ordinal(h['position'])} in line"
    return f"{at} · {h['state']}"


def ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def budget_left(me: dict) -> str:
    """" · $0.98 of $1.00 left": what is left of the participant's model budget, from the lab's /api/me; empty when
    not connected or the lab does not say."""
    left = me.get("left")
    left = left.get("models") if isinstance(left, dict) else None
    if isinstance(left, bool) or not isinstance(left, (int, float)):
        return ""
    return f" · ${left:.2f} of ${lab.AGENT_BUDGET_USD:.2f} left"


# ---- connect, new

def cmd_update(args) -> None:
    """The newest lab code: the setup script's clone step only (your edits and notebook are kept), in seconds."""
    if not lab.LAB.exists():
        raise Fail("not connected to the lab; run the setup line from your lab page")
    env = dict(os.environ, LAB_URL=json.loads(lab.LAB.read_text())["url"], LAB_SETUP_STEPS="base")
    os.execvpe("bash", ["bash", "-c", 'curl -fsSL "$LAB_URL/setup" | bash'], env)


def cmd_page(args) -> None:
    """A link that signs a browser in to your lab page: a new browser, or one that lost its login."""
    if not lab.LAB.exists():
        raise Fail("not connected to the lab; run the setup line from your lab page")
    link = lab.call("/api/login-link", {})
    console.print(f"open within {link['minutes']} minutes: {link['url']}")


def cmd_connect(args) -> None:
    ok(f"connected as [num]{lab.connect(args.url, args.token)}[/]"
       + (" · lab serve starts the agent model (about 4 min)" if (VENV / "bin" / "vllm").exists() else ""))


# ---- vespa

def cmd_vespa(args) -> None:
    if args.folder and args.action == "down":
        raise Fail("lab vespa down takes no search folder")
    {"up": vespa_up, "deploy": vespa_deploy, "down": vespa_down, "status": vespa_status}[args.action](args.folder)


def vespa_up(folder: str | None) -> None:
    start = time.time()
    with progress(TextColumn("{task.completed:,.0f} documents")) as bar:
        step = bar.add_task("starting Vespa", total=None)

        def feeding(name: str):
            bar.update(step, visible=False)
            task = bar.add_task(f"feeding {name}", total=None)
            return lambda n: bar.advance(task, n)

        result = labvespa.up(folder, lambda m: bar.update(step, description=m), feeding, warn)
    show_changes(result["deploy"])
    fed = " · ".join(f"fed {name} in {s:.0f} s" for name, s in result["fed"].items()) or "the data was fed before"
    ok(f"vespa up at {labvespa.QUERY_URL} · {fed} · {time.time() - start:.0f} s")


def vespa_deploy(folder: str | None) -> None:
    path = labvespa.search_folder(folder)
    with console.status("deploying") as status:
        answer = labvespa.deploy(path, lambda m: status.update(m), warn)
    show_changes(answer)
    ok(f"deployed {labvespa.shown(path / 'app')}")
    unfed = labvespa.unfed(path)
    if unfed:
        warn(f"{' and '.join(unfed)}: {'; '.join(sorted(set(unfed.values())))}; lab vespa up {folder or ''}".rstrip()
             + " feeds them (about 1.5 minutes)")


def vespa_sync(folder: pathlib.Path) -> None:
    """Deploys folder's app/ when it changed since the last deploy, in one line; fails, naming lab vespa up, when the
    documents need feeding again."""
    start = time.time()
    with console.status("checking Vespa") as status:
        answer = labvespa.sync(folder, lambda m: status.update(m), warn)
    if answer is not None:
        show_changes(answer)
        ok(f"deployed {labvespa.shown(folder / 'app')}, which changed since the last deploy · {time.time() - start:.0f} s")


def show_changes(answer: dict) -> None:
    """Vespa's warnings on a deploy, and what it needs restarted, refed or reindexed."""
    lines = labvespa.changes(answer)
    for line in lines:
        warn(line)
    if any(line.startswith("restart") for line in lines):
        warn(f"to restart: docker restart {labvespa.NAME}, then lab vespa up")
    if any(line.startswith("reindex") for line in lines):
        warn("vespa reindexes the documents in the background")


def warn(message: str) -> None:
    console.print(Text.assemble(("! ", "warn"), message))


def vespa_down(folder: str | None) -> None:
    ok(f"removed {labvespa.NAME} and its data" if labvespa.down() else f"no {labvespa.NAME} container")


def vespa_status(folder: str | None) -> None:
    with console.status("checking"):
        s = labvespa.status(folder)
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
    s.add_argument("--force", action="store_true", help="continue when other programs use the GPU's memory")
    s.set_defaults(run=cmd_setup)
    s = sub.add_parser("connect", help="save the lab URL and your token (both on your lab page)")
    s.add_argument("url")
    s.add_argument("token")
    s.set_defaults(run=cmd_connect)
    s = sub.add_parser("serve", help="start the agent model (vLLM) in the background")
    s.add_argument("--stop", action="store_true", help="stop it")
    s.add_argument("--timeout", type=int, default=900, help="seconds to wait for it (default 900)")
    s.set_defaults(run=cmd_serve)
    s = sub.add_parser("update", help="the newest lab code, in seconds (your edits and notebook are kept)")
    s.set_defaults(run=cmd_update)
    s = sub.add_parser("page", help="a link that signs a browser in to your lab page")
    s.set_defaults(run=cmd_page)
    s = sub.add_parser("status", help="GPU, vLLM, lab connection, data and models")
    s.set_defaults(run=cmd_status)
    s = sub.add_parser("vespa", help="your own Vespa in Docker: up, deploy, down, status")
    s.add_argument("action", choices=("up", "deploy", "down", "status"))
    s.add_argument("folder", nargs="?", help="the search folder, with app/ and feed.py (default: ./search, else the lab's)")
    s.set_defaults(run=cmd_vespa)
    s = sub.add_parser("eval", help="score your search or agent on the dev questions and save the run")
    s.add_argument("file", help="search (search/search.py), agent (agent.py) or a file")
    s.add_argument("--worst", type=int, default=0, metavar="N", help="also show the N worst questions")
    s.add_argument("--data", choices=[name for name, _ in EVAL_SETS], help="only this dataset (search files)")
    s.set_defaults(run=cmd_eval)
    s = sub.add_parser("submit", help="send ./search (search, or nothing) or agent.py for grading on hidden questions "
                       "(the latest counts), and its latest saved lab eval's dev score to the live board")
    s.add_argument("file", nargs="?", help="search (./search), agent (agent.py) or a file")
    s.add_argument("-m", metavar="LABEL", help="a few words on what it does (default: its run name)")
    s.set_defaults(run=cmd_submit)
    s = sub.add_parser("runs", help="your saved runs, newest first")
    s.add_argument("what", nargs="?", metavar="search|agent|FILE", help="whose runs (default search)")
    s.add_argument("-n", type=int, default=20, help="how many (default 20)")
    s.set_defaults(run=cmd_runs)
    s = sub.add_parser("compare", help="what changed between two runs (default: the last two of the same file), or in "
                       "one question: lab compare question NUMBER")
    s.add_argument("words", nargs="*", metavar="[search|agent] [question [DATASET] NUMBER-OR-WORDS] [RUN RUN]")
    s.add_argument("-n", type=int, default=5, help="questions shown each way (default 5)")
    s.set_defaults(run=cmd_compare)
    s = sub.add_parser("trace", help="one agent question step by step: its tool calls, what they returned, its answer")
    s.add_argument("words", nargs="+", metavar="[crm] NUMBER-OR-WORDS [RUN]",
                   help="the question (its number or words of it) and the run (default: the newest agent run; NAME#2 is "
                        "the second eval of NAME)")
    s.add_argument("--json", action="store_true", help="the saved trajectory as JSON")
    s.set_defaults(run=cmd_trace)
    s = sub.add_parser("query", help="ask your search or agent any question: lab query search \"...\", lab query agent \"...\"")
    s.add_argument("what", choices=("search", "agent"))
    s.add_argument("text", nargs="+", metavar="QUESTION")
    s.add_argument("--data", choices=[name for name, _ in EVAL_SETS], help="the search's dataset (default fiqa)")
    s.add_argument("-k", type=int, default=10, help="how many documents (default 10)")
    s.set_defaults(run=cmd_query)
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
