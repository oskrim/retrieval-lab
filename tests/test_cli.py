"""pytest -p no:xdist tests"""

import json
import os
import pathlib
import sys
import textwrap
import threading

import numpy as np
import pytest
from rich.cells import cell_len

import cli

FAKE_ENDPOINT = ("http://localhost:8000/v1/chat/completions", {}, "fake-model")


def write(tmp_path: pathlib.Path, name: str, code: str) -> pathlib.Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(code))
    return path


def test_check_rows_accepts_k_distinct_ints_in_range():
    assert cli.check_rows(np.array([3, 1, 2]), 5, 3) == [3, 1, 2]
    assert cli.check_rows([np.int64(0), 4], 5, 2) == [0, 4]


@pytest.mark.parametrize("rows, why", [
    ([1, 2], "returned 2 rows"),
    ([1, 1, 2], "same row twice"),
    ([1, 2, 5], "out of range"),
    ([1, -1, 2], "out of range"),
    ([1.0, 2.0, 3.0], "rows are ints"),
    ([True, 2, 3], "rows are ints"),
    (["a", "b", "c"], "rows are ints"),
    (None, "return a list or array"),
    ({1, 2, 3}, "has no order"),
])
def test_check_rows_refuses(rows, why):
    with pytest.raises(cli.ContractError, match=why):
        cli.check_rows(rows, 5, 3)


def test_check_deals():
    assert cli.check_deals({"D0001", "D0002"}) == {"D0001", "D0002"}
    assert cli.check_deals(set()) == set()
    with pytest.raises(cli.ContractError, match="return a set"):
        cli.check_deals(["D0001"])
    with pytest.raises(cli.ContractError, match="deal ids are str"):
        cli.check_deals({1})


def test_file_kind(tmp_path):
    kinds = {name: code for name, code in (("s.py", "def search(ds, q, k=10): return []"), ("a.py", "def answer(q): return set()"),
                                           ("both.py", "def search(): pass\ndef answer(): pass"), ("none.py", "x = 1"))}
    assert cli.file_kind(cli.load_file(str(write(tmp_path, "s.py", kinds["s.py"])))[1]) == "search"
    assert cli.file_kind(cli.load_file(str(write(tmp_path, "a.py", kinds["a.py"])))[1]) == "agent"
    for name in ("both.py", "none.py"):
        with pytest.raises(cli.Fail):
            cli.file_kind(cli.load_file(str(write(tmp_path, name, kinds[name])))[1])


def test_errors_show_only_the_participants_frames(tmp_path):
    path = write(tmp_path, "mine.py", """\
        import json

        def search(ds, question, k=10):
            return json.loads(question)
    """)
    _, module = cli.load_file(str(path))
    run = cli.checked_search(module, path)
    with pytest.raises(cli.UserCodeError) as info:
        run(None, "not json", 10)
    frames = cli.user_frames(info.value.exc, path)
    assert [(f.name, f.lineno, f.line) for f in frames] == [("search", 4, "return json.loads(question)")]
    assert len(cli.traceback.extract_tb(info.value.exc.__traceback__)) > 1  # json's own frames were left out


def test_syntax_errors_point_at_the_line(tmp_path):
    path = write(tmp_path, "broken.py", "def search(ds, question, k=10):\n    return [1, 2\n")
    with pytest.raises(cli.UserCodeError) as info:
        cli.load_file(str(path))
    assert [f.filename for f in cli.user_frames(info.value.exc, path)] == [str(path)]


def test_main_exits_nonzero_without_a_cli_traceback(tmp_path, capsys, monkeypatch):
    path = write(tmp_path, "boom.py", "def answer(question):\n    raise ValueError('boom')\n")
    monkeypatch.setattr(cli, "require_agent", lambda: FAKE_ENDPOINT)
    monkeypatch.setattr(cli, "HOME", tmp_path)
    assert cli.main(["eval", str(path)]) == 1
    out = capsys.readouterr().out
    assert "boom.py:2 in answer" in out and "ValueError: boom" in out and "cli.py" not in out


def test_sys_exit_in_the_participants_code_is_their_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    at_import = write(tmp_path, "quits.py", "import sys\nsys.exit('bye')\n")
    assert cli.main(["eval", str(at_import)]) == 1
    assert "quits.py raised SystemExit" in capsys.readouterr().out
    path = write(tmp_path, "exits.py", "import sys\n\ndef answer(question):\n    sys.exit(0)\n")
    _, module = cli.load_file(str(path))
    with pytest.raises(cli.UserCodeError) as info:
        cli.checked_answer(module, path)("q")
    assert [f.lineno for f in cli.user_frames(info.value.exc, path)] == [4]


def test_errors_in_a_helper_module_beside_the_file_show_its_line(tmp_path, capsys):
    write(tmp_path, "helpers.py", "def rows(k):\n    return list(range(undefined_k))\n")
    path = write(tmp_path, "mine.py", "from helpers import rows\n\ndef search(ds, question, k=10):\n    return rows(k)\n")
    _, module = cli.load_file(str(path))
    with pytest.raises(cli.UserCodeError) as info:
        cli.checked_search(module, path)(None, "q", 10)
    cli.show_user_error(info.value)
    out = capsys.readouterr().out
    assert "mine.py:4 in search" in out and "helpers.py:2 in rows" in out and "lab.py" not in out


@pytest.fixture
def evals(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "EVALS", tmp_path / "evals.jsonl")
    return cli.EVALS


def ev(at, file, score, dataset="fiqa dev", kind="search", **more):
    return dict(at=at, file=file, kind=kind, dataset=dataset, score=score, **more)


def test_eval_history_skips_lines_that_are_not_evals(evals):
    cli.remember(pathlib.Path("/w/a.py"), "search", 1.0, cli.Result("fiqa dev", {"q1": 1.0, "q2": 0.0}, 0.5, 0.4, 0.6))
    legacy = ev(2, "/w/b.py", 0.4)  # written before evals.jsonl kept the data version
    with open(evals, "a") as f:
        for line in ['{"at": 1, "file": "b.py", "kind": "sea', "3", "[]", json.dumps(ev(3, "/w/c.py", "high")),
                     json.dumps(dict(at=4, file="/w/d.py", kind="search", dataset="fiqa dev")),
                     json.dumps(ev(5, "/w/e.py", 0.9, data="data-v1")), json.dumps(legacy), "\x00\xff"]:
            f.write(line + "\n")
    rows = cli.history()
    assert [r["file"] for r in rows] == ["/w/a.py", "/w/b.py"]
    assert rows[0]["per"] == {"q1": 1.0, "q2": 0.0} and rows[0]["questions"] == 2 and rows[0]["run"] == 1.0


def test_previous_is_the_latest_eval_of_the_file_and_best_the_highest_of_any_file():
    rows = [ev(1, "/w/a.py", 0.50), ev(2, "/w/b.py", 0.60), ev(3, "/w/a.py", 0.40), ev(4, "/w/c.py", 0.60),
            ev(5, "/w/a.py", 0.90, dataset="caiq test"), ev(6, "/w/a.py", 0.99, kind="agent", dataset="crm dev")]
    assert cli.previous(rows, "search", "fiqa dev", "/w/a.py")["at"] == 3
    assert cli.previous(rows, "search", "fiqa dev", "/w/new.py") is None
    assert cli.best(rows, "search", "fiqa dev")["at"] == 2  # the earliest of two equal bests
    assert cli.best(rows, "search", "caiq test")["at"] == 5
    assert cli.best(rows, "agent", "fiqa dev") is None


def test_flips_count_the_questions_that_scored_higher_and_lower():
    before = ev(1, "/w/a.py", 0.5, per={"a": 0.5, "b": 0.2, "c": 1.0, "gone": 0.3})
    assert cli.flips(before, {"a": 0.7, "b": 0.2, "c": 0.0, "new": 1.0}) == (1, 1)
    assert cli.flips(ev(1, "/w/a.py", 0.5), {"a": 1.0}) is None  # an eval from before per-question scores
    assert cli.flips(None, {"a": 1.0}) is None


def test_recent_runs_group_a_runs_datasets_oldest_first():
    rows = [ev(10.1, "/w/a.py", 0.5), ev(10.2, "/w/a.py", 0.7, dataset="caiq test"),  # one run without a run id
            ev(20.0, "/w/a.py", 0.6, run=20.0), ev(20.5, "/w/a.py", 0.8, dataset="caiq test", run=20.0),
            ev(30.0, "/w/b.py", 0.55, run=30.0), ev(31.0, "/w/a.py", 0.9, kind="agent", dataset="crm dev", run=31.0)]
    runs = cli.recent_runs(rows, "search", 5)
    assert [(r["file"], scores) for r, scores in runs] == [
        ("/w/a.py", {"fiqa dev": 0.5, "caiq test": 0.7}), ("/w/a.py", {"fiqa dev": 0.6, "caiq test": 0.8}),
        ("/w/b.py", {"fiqa dev": 0.55})]
    assert [r["at"] for r, _ in cli.recent_runs(rows, "search", 2)] == [20.0, 30.0]


def test_sparkline():
    assert cli.sparkline([0.1, 0.5, 0.3]) == "▁█▄"
    assert cli.sparkline([0.5, 0.5]) == "▄▄"


def test_the_result_screen_fits_80_columns(evals):
    rows = [ev(1, "/w/search.py", 0.55, run=1, per={"q1": 0.5, "q2": 1.0}),
            ev(1, "/w/search.py", 0.70, dataset="caiq test", run=1),
            ev(2, "/w/a_rather_long_file_name_for_search.py", 0.60, run=2)]
    results = [cli.result("fiqa dev", {"q1": 0.25, "q2": 1.0}), cli.result("caiq test", {"q9": 1.0})]
    out = cli.Console(width=80, record=True, file=open(os.devnull, "w"))
    out.print(cli.runs_table("search", cli.EVAL_LABELS, rows))
    out.print(cli.result_table(pathlib.Path("/w/search.py"), "search", "nDCG@10", results, rows, 3.2))
    text = out.export_text()
    assert max(map(cell_len, text.splitlines())) <= 80
    assert "↑0 ↓1" in text and "★" in text and "0.600 " in text and "a_rather" in text
    assert "0.550" in text and "–" in text  # the previous eval of search.py; no flips without per-question scores


def test_agent_eval_marks_the_questions_that_flipped(tmp_path, evals, capsys, monkeypatch):
    monkeypatch.setattr(cli, "require_agent", lambda: FAKE_ENDPOINT)
    monkeypatch.setattr(cli.lab, "crm_dev", lambda: [("first question", {"D0001"}), ("second question", {"D0002"})])
    path = tmp_path / "agent.py"
    for found in ("{'D0001'}", "{'D0002'}  # a size of its own: Python checks the cached bytecode by size and second"):
        path.write_text(f"def answer(question):\n    return {found}\n")
        assert cli.main(["eval", str(path)]) == 0
    out = capsys.readouterr().out.split("first question")[-1]
    assert "✓ ↑ second question" in out and "↑1 ↓1" in out and "fake-model" in out
    assert [r["score"] for r in cli.history()] == [0.5, 0.5]


@pytest.mark.parametrize("label", ["", "   ", "x" * (cli.LABEL_CHARS + 1)])
def test_submit_refuses_a_bad_label_before_running_anything(tmp_path, capsys, monkeypatch, label):
    monkeypatch.setattr(cli, "load_file", lambda name: pytest.fail("the file must not run"))
    assert cli.main(["submit", "search.py", "-m", label]) == 1
    assert f"1 to {cli.LABEL_CHARS} characters" in capsys.readouterr().out


def test_new_says_why_it_cannot_write(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "search.py").symlink_to(tmp_path / "missing" / "x.py")
    assert cli.main(["new", "search"]) == 1
    assert "search.py exists" in capsys.readouterr().out
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o555)
    monkeypatch.chdir(locked)
    assert cli.main(["new", "agent"]) == 1
    assert "cannot write" in capsys.readouterr().out


def test_serve_shows_the_log_when_vllm_exits_while_starting(tmp_path, capsys, monkeypatch):
    for name, value in (("HOME", tmp_path), ("VENV", tmp_path / "venv"), ("SERVE_LOG", tmp_path / "serve.log"),
                        ("SERVE_PID", tmp_path / "serve.pid")):
        monkeypatch.setattr(cli, name, value)
    monkeypatch.setattr(cli.lab, "local_agent_model", lambda: None)
    monkeypatch.setattr(cli, "vllm_args", lambda: ["--gpu-memory-utilization", "0.8"])
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    vllm = tmp_path / "venv" / "bin" / "vllm"
    vllm.write_text("#!/bin/sh\necho 'CUDA error: no kernel image'\nexit 1\n")
    vllm.chmod(0o755)
    assert cli.main(["serve", "--timeout", "30"]) == 1
    out = capsys.readouterr().out
    assert "vLLM stopped while starting" in out and "CUDA error" in out
    assert not cli.SERVE_PID.exists()


def test_ctrl_c_does_not_wait_for_worker_threads(capsys, monkeypatch):
    release = threading.Event()
    exits = []

    def run(args):
        threading.Thread(target=release.wait, args=(30,)).start()  # an agent thread waiting on the model
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "cmd_board", run)
    monkeypatch.setattr(cli.os, "_exit", exits.append)
    try:
        cli.main(["board"])
    finally:
        release.set()
    assert exits == [130] and "stopped" in capsys.readouterr().out


def test_model_errors_are_not_blamed_on_the_participant(tmp_path, capsys, monkeypatch):
    path = write(tmp_path, "agent.py", """\
        import lab

        def answer(question):
            raise lab.ModelError("the agent model Qwen/Qwen3.5-9B answered 500: CUDA out of memory")
    """)
    monkeypatch.setattr(cli, "require_agent", lambda: FAKE_ENDPOINT)
    monkeypatch.setattr(cli, "HOME", tmp_path)
    assert cli.main(["eval", str(path)]) == 1
    out = capsys.readouterr().out
    assert "answered 500: CUDA out of memory" in out and "raised" not in out


def test_vllm_args_fill_the_snapshot_and_leave_memory_for_the_reranker(monkeypatch):
    import huggingface_hub
    import torch

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, local_files_only: "/models/nemo")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (80e9, 80e9))
    args = cli.vllm_args()
    assert "/models/nemo/nano_v3_reasoning_parser.py" in args and not any("{snapshot}" in a for a in args)
    assert args[-2:] == ["--gpu-memory-utilization", str(round(1 - cli.lab.VLLM_LEAVE_GB * 1e9 / 80e9, 2))]
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (24e9, 24e9))
    with pytest.raises(cli.Fail):
        cli.vllm_args()


def test_new_vespa_copies_the_app_search_and_nick_and_keeps_what_is_there(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "my_nick", lambda: "ada")
    assert cli.main(["new", "vespa"]) == 0
    assert (tmp_path / "vespa" / "app" / "schemas" / "fiqa.sd").read_text() == (cli.labvespa.APP / "schemas" / "fiqa.sd").read_text()
    assert (tmp_path / "vespa" / "search.py").read_text() == (cli.TEMPLATES / "vespa" / "search.py").read_text()
    assert (tmp_path / "vespa" / "NICK").read_text().splitlines()[1:] == ["ada"]
    (tmp_path / "vespa" / "app" / "services.xml").write_text("mine")
    (tmp_path / "vespa" / "NICK").unlink()
    assert cli.main(["new", "vespa"]) == 0
    assert (tmp_path / "vespa" / "app" / "services.xml").read_text() == "mine"
    assert "wrote vespa/NICK · kept app, search.py" in capsys.readouterr().out
    assert cli.main(["new", "vespa"]) == 1
    assert "has app, search.py and NICK already" in capsys.readouterr().out


@pytest.mark.parametrize("argv, called", [
    (["vespa", "up"], ("up", None)),
    (["vespa", "up", "fork/vespa/app"], ("up", "fork/vespa/app")),
    (["vespa", "deploy", "vespa/app"], ("deploy", "vespa/app")),
    (["vespa", "down"], ("down", None)),
    (["vespa", "status"], ("status", None)),
])
def test_vespa_commands(argv, called, monkeypatch):
    seen = []
    for action in ("up", "deploy", "down", "status"):
        monkeypatch.setattr(cli, f"vespa_{action}", lambda app, action=action: seen.append((action, app)))
    assert cli.main(argv) == 0
    assert seen == [called]


def test_vespa_down_and_status_take_no_app(capsys):
    assert cli.main(["vespa", "down", "vespa/app"]) == 1
    assert "takes no application directory" in capsys.readouterr().out


def test_vespa_errors_are_not_blamed_on_the_participant(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli.labvespa, "query", lambda *a, **k: (_ for _ in ()).throw(cli.labvespa.VespaError("Vespa does not answer")))
    path = write(tmp_path, "s.py", """
        import labvespa
        def search(ds, question, k=10):
            return labvespa.query(ds.name, "select * from fiqa where true", "hybrid")
    """)
    monkeypatch.setattr(cli.lab, "load", lambda name, split="test": cli.lab.Dataset(name, ["a"], [""], [""], {"q": "x"}, {"q": {"a": 1}}, {}))
    assert cli.main(["eval", str(path)]) == 1
    out = capsys.readouterr().out
    assert "Vespa does not answer" in out and "s.py raised" not in out
