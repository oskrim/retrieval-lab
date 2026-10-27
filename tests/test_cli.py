"""pytest -p no:xdist tests"""

import json
import os
import re
import pathlib
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
    assert cli.file_kind(cli.load_file(str(write(tmp_path, "both.py", kinds["both.py"])))[1]) == "agent"  # a search tool
    with pytest.raises(cli.Fail):
        cli.file_kind(cli.load_file(str(write(tmp_path, "none.py", kinds["none.py"])))[1])


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
    cli.remember(pathlib.Path("/w/a.py"), "search", 1.0, cli.Result("fiqa dev", {"q1": 1.0, "q2": 0.0}, 0.5, 0.4, 0.6),
                 "ab12", "brave-otter-42")
    legacy = ev(2, "/w/b.py", 0.4)  # written before evals.jsonl kept the data version
    with open(evals, "a") as f:
        for line in ['{"at": 1, "file": "b.py", "kind": "sea', "3", "[]", json.dumps(ev(3, "/w/c.py", "high")),
                     json.dumps(dict(at=4, file="/w/d.py", kind="search", dataset="fiqa dev")),
                     json.dumps(ev(5, "/w/e.py", 0.9, data="data-v1")), json.dumps(legacy), "\x00\xff"]:
            f.write(line + "\n")
    rows = cli.history()
    assert [r["file"] for r in rows] == ["/w/a.py", "/w/b.py"]
    assert rows[0]["questions"] == 2 and rows[0]["run"] == 1.0
    assert rows[0]["name"] == "brave-otter-42" and "name" not in rows[1]


def test_files_are_told_apart_by_their_path_in_the_repo(tmp_path):
    for clone in ("one", "two"):
        (tmp_path / clone / ".git").mkdir(parents=True)
    assert cli.file_key(str(tmp_path / "one" / "search" / "search.py")) == "search/search.py"
    assert cli.file_key(str(tmp_path / "two" / "search" / "search.py")) == "search/search.py"
    assert cli.file_key(str(tmp_path / "one" / "search.py")) == "search.py"
    assert cli.file_key("/outside/mine.py") == "/outside/mine.py"


def test_the_result_screen_fits_80_columns(tmp_path):
    (tmp_path / ".git").mkdir()
    mine = tmp_path / "search" / "search.py"
    results = [cli.result("fiqa dev", {"q1": 0.25, "q2": 1.0}), cli.result("caiq test", {"q9": 1.0})]
    out = cli.Console(width=80, record=True, file=open(os.devnull, "w"))
    out.print(cli.result_table(mine, "nDCG@10", results, 3.2))
    text = out.export_text()
    assert max(map(cell_len, text.splitlines())) <= 80
    assert "search/search.py · nDCG@10 · 3.2 s" in text and "0.625" in text and "1.000" in text


def test_agent_eval_marks_each_question_and_saves_the_run(tmp_path, evals, capsys, monkeypatch):
    monkeypatch.setattr(cli, "require_agent", lambda: FAKE_ENDPOINT)
    monkeypatch.setattr(cli.lab, "crm_dev", lambda: [("first question", {"D0001"}), ("second question", {"D0002"})])
    path = tmp_path / "agent.py"
    path.write_text("def answer(question):\n    return {'D0002'}\n")
    assert cli.main(["eval", str(path)]) == 0
    out = capsys.readouterr().out
    assert "✗ crm 1  first question" in out and "✓ crm 2  second question" in out and "fake-model" in out
    assert [r["score"] for r in cli.history()] == [0.5]  # numbered as lab trace crm 1 and lab compare question crm 1


@pytest.mark.parametrize("label", ["", "   ", "x" * (cli.LABEL_CHARS + 1)])
def test_submit_refuses_a_bad_label_before_running_anything(tmp_path, capsys, monkeypatch, label):
    monkeypatch.setattr(cli, "load_file", lambda name: pytest.fail("the file must not run"))
    assert cli.main(["submit", "agent.py", "-m", label]) == 1
    assert f"1 to {cli.LABEL_CHARS} characters" in capsys.readouterr().out


def test_a_search_file_outside_the_search_folder_goes_to_the_live_board_only(connected, evals, capsys, monkeypatch):
    monkeypatch.setattr(cli, "send_handin", lambda data, route: pytest.fail("only search/ is uploaded"))
    path = write(connected, "mine.py", "def search(ds, question, k=10):\n    return list(range(k))\n")
    assert cli.main(["submit", str(path)]) == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "only search/ is graded for the final board, so mine.py goes to the live board only" in out


def test_serve_shows_the_log_when_vllm_exits_while_starting(tmp_path, capsys, monkeypatch):
    for name, value in (("HOME", tmp_path), ("VENV", tmp_path / "venv"), ("SERVE_LOG", tmp_path / "serve.log"),
                        ("SERVE_PID", tmp_path / "serve.pid")):
        monkeypatch.setattr(cli, name, value)
    monkeypatch.setattr(cli.lab, "local_agent_model", lambda: None)
    monkeypatch.setattr(cli, "gpu", lambda: ("NVIDIA A100 80GB", 80 * 10**9, 80 * 10**9))
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

    monkeypatch.setattr(cli, "cmd_status", run)
    monkeypatch.setattr(cli.os, "_exit", exits.append)
    try:
        cli.main(["status"])
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


def search_folder(root: pathlib.Path) -> pathlib.Path:
    folder = root / "search"
    (folder / "app" / "__pycache__").mkdir(parents=True)
    (folder / "search.py").write_text("def search(ds, question, k=10): ...\n")
    (folder / "feed.py").write_text("def documents(ds, fields): ...\n")
    (folder / "analyze.py").write_text("")
    (folder / "helper.py").write_text("X = 1\n")
    (folder / "app" / "schema.sd").write_text("schema dataset {}\n")
    (folder / "app" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\0")
    (folder / "old.pyc").write_bytes(b"\0")
    (folder / ".env").write_text("SECRET=1\n")
    (folder / ".ipynb_checkpoints").mkdir()
    (folder / ".ipynb_checkpoints" / "search-checkpoint.py").write_text("")
    (folder / "app" / "link.sd").symlink_to(folder / "app" / "schema.sd")
    return folder


def unpacked(data: bytes) -> dict[str, bytes]:
    with cli.tarfile.open(fileobj=cli.io.BytesIO(data), mode="r:gz") as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers()}


def test_pack_keeps_the_participants_files_and_leaves_out_caches_dotfiles_links_and_big_files(tmp_path, monkeypatch):
    folder = search_folder(tmp_path)
    monkeypatch.setattr(cli, "HANDIN_FILE_BYTES", 100)
    (folder / "app" / "big.json").write_text("x" * 101)
    data, packed, left_out = cli.pack(folder)
    assert packed == ["analyze.py", "app/schema.sd", "feed.py", "helper.py", "search.py"]
    assert left_out == ["app/big.json (0.0 MB)", "app/link.sd (a link)"]
    assert unpacked(data) == {"analyze.py": b"", "app/schema.sd": b"schema dataset {}\n",
                              "feed.py": b"def documents(ds, fields): ...\n", "helper.py": b"X = 1\n",
                              "search.py": b"def search(ds, question, k=10): ...\n"}
    os.utime(folder / "search.py", (1, 1))
    assert cli.pack(folder)[0] == data  # the same files pack to the same bytes


@pytest.fixture
def connected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.lab, "LAB", tmp_path / ".lab.json")
    cli.lab.LAB.write_text(json.dumps(dict(url="http://lab.test", token="maria.tok")))
    return tmp_path


def test_submit_search_sends_the_folder_and_its_latest_dev_score(connected, evals, capsys, monkeypatch):
    folder = search_folder(connected)
    sent, posted = [], []
    monkeypatch.setattr(cli, "send_handin", lambda data, route: sent.append((route, data)) or dict(
        nick="maria", at=cli.time.time(), files=5, bytes=len(data), sha256="ab" * 32))
    monkeypatch.setattr(cli.lab, "call", lambda path, body: posted.append((path, body)) or dict(score=body["scores"]["fiqa dev"], rank=2, of=5))
    assert cli.main(["submit"]) == 1
    out = " ".join(capsys.readouterr().out.split())
    assert "no saved fiqa dev eval of search.py as it is now; lab eval search first" in out and sent == posted == []
    path = folder / "search.py"
    run_id = cli.config_id(path)
    for at, dataset, score in ((1.0, "fiqa dev", 0.3), (2.0, "caiq test", 0.6), (3.0, "fiqa dev", 0.4)):
        cli.remember(path, "search", at, cli.Result(dataset, {"q": score}, score, score, score), run_id, "brave-otter-42")
    cli.remember(path, "search", 4.0, cli.Result("fiqa dev", {"q": 0.9}, 0.9, 0.9, 0.9), "0" * 16, "other-run-1")  # another version
    assert cli.main(["submit"]) == 0
    out = " ".join(capsys.readouterr().out.split())
    assert sent[0][0] == "/api/handin" and set(unpacked(sent[0][1])) == {"analyze.py", "app/schema.sd", "feed.py",
                                                                   "helper.py", "search.py"}
    assert "submitted search/" in out and "5 files" in out and "left out search/app/link.sd (a link)" in out
    assert "graded in a few minutes · lab status shows it" in out and "next:" not in out
    posted.clear()
    assert cli.main(["submit", "search", "-m", "bm25 + colbert"]) == 0  # the same as lab submit
    assert len(sent) == 2 and posted == [("/api/board", dict(label="bm25 + colbert", scores={"fiqa dev": 0.4, "caiq test": 0.6}))]
    assert "live board: fiqa dev 0.4000 (bm25 + colbert) · rank 2 of 5" in " ".join(capsys.readouterr().out.split())


def test_submit_agent_sends_the_file_and_its_latest_dev_score(connected, evals, capsys, monkeypatch):
    path = write(connected, "agent.py", "def answer(question):\n    return set()\n")
    sent, posted = [], []
    monkeypatch.setattr(cli, "send_handin", lambda data, route: sent.append((route, data)) or dict(at=cli.time.time()))
    monkeypatch.setattr(cli.lab, "call", lambda path, body: posted.append((path, body)) or dict(score=0.5, rank=1, of=1))
    monkeypatch.setattr(cli, "require_agent", lambda: pytest.fail("no model at submit"))
    cli.remember(path.resolve(), "agent", 1.0, cli.Result("crm dev", {"q": 0.5}, 0.5, 0.5, 0.5), cli.config_id(path.resolve()), "x-y-1")
    assert cli.main(["submit", "agent"]) == 0
    assert sent == [("/api/agent-handin", path.read_bytes())]
    assert posted == [("/api/agent-board", dict(label=cli.run_name(cli.config_id(path.resolve()), {"x-y-1": cli.config_id(path.resolve())}),
                                                  scores={"crm dev": 0.5}))]
    assert "graded in a few minutes · lab status shows it" in " ".join(capsys.readouterr().out.split())
    path.write_text("def answer(question):\n    return {'D0001'}\n")  # changed since its eval
    assert cli.main(["submit", "agent"]) == 1
    assert len(sent) == 1 and len(posted) == 1 and "no saved crm dev eval of agent.py" in " ".join(capsys.readouterr().out.split())


def test_submit_search_says_what_to_do_when_not_connected_or_without_the_folder(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.lab, "LAB", tmp_path / ".lab.json")
    monkeypatch.setattr(cli, "send_handin", lambda data, route: pytest.fail("sent"))
    assert cli.main(["submit"]) == 1
    assert "not connected to the lab; lab connect URL TOKEN (both on your lab page)" in capsys.readouterr().out
    cli.lab.LAB.write_text(json.dumps(dict(url="http://lab.test", token="t")))
    assert cli.main(["submit"]) == 1
    out = " ".join(capsys.readouterr().out.split())
    assert f"no search/search.py here; run lab in {cli.lab.HERE}, which has the lab's" in out
    search_folder(tmp_path)
    (tmp_path / "search" / "feed.py").unlink()
    assert cli.main(["submit"]) == 1
    out = " ".join(capsys.readouterr().out.split())
    assert "no search/feed.py in" in out and f"run lab in {cli.lab.HERE}" in out


def test_submit_search_refuses_a_folder_over_the_limit(connected, capsys, monkeypatch):
    folder = search_folder(connected)
    (folder / "app" / "noise.bin").write_bytes(os.urandom(4000))
    monkeypatch.setattr(cli, "HANDIN_BYTES", 2000)
    monkeypatch.setattr(cli, "send_handin", lambda data, route: pytest.fail("sent"))
    assert cli.main(["submit"]) == 1
    assert "Move data and large models out" in capsys.readouterr().out


def test_submit_search_shows_the_labs_refusal(connected, capsys, monkeypatch):
    search_folder(connected)

    def refuse(req, timeout):
        raise cli.urllib.error.HTTPError(req.full_url, 400, "Bad Request", {},
                                         cli.io.BytesIO(b'{"detail": "no app/ at the top"}'))

    monkeypatch.setattr(cli.urllib.request, "urlopen", refuse)
    monkeypatch.setattr(cli, "require_eval", lambda kind, path: None)
    assert cli.main(["submit"]) == 1
    assert "the lab says: no app/ at the top" in capsys.readouterr().out


@pytest.mark.parametrize("argv, called", [
    (["vespa", "up"], ("up", None)),
    (["vespa", "up", "fork/search"], ("up", "fork/search")),
    (["vespa", "deploy", "search"], ("deploy", "search")),
    (["vespa", "down"], ("down", None)),
    (["vespa", "status"], ("status", None)),
    (["vespa", "status", "fork/search"], ("status", "fork/search")),
])
def test_vespa_commands(argv, called, monkeypatch):
    seen = []
    for action in ("up", "deploy", "down", "status"):
        monkeypatch.setattr(cli, f"vespa_{action}", lambda folder, action=action: seen.append((action, folder)))
    assert cli.main(argv) == 0
    assert seen == [called]


def test_vespa_down_takes_no_folder(capsys):
    assert cli.main(["vespa", "down", "search"]) == 1
    assert "takes no search folder" in capsys.readouterr().out


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


def test_serve_without_a_big_gpu_says_the_agent_runs_through_the_api(capsys, monkeypatch):
    monkeypatch.setattr(cli.lab, "local_agent_model", lambda: None)
    monkeypatch.setattr(cli, "serve_pid", lambda: None)
    monkeypatch.setattr(cli, "gpu", lambda: ("NVIDIA GeForce RTX 3090", 24 * 10**9, 25 * 10**9))
    assert cli.main(["serve"]) == 0
    out = capsys.readouterr().out
    assert "nothing to serve" in out and "through the API" in out
    monkeypatch.setattr(cli, "gpu", lambda: None)
    assert cli.main(["serve"]) == 0
    assert "no GPU here" in capsys.readouterr().out


def test_search_and_agent_name_the_usual_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(cli.Fail, match="no search/search.py here; run lab in .*, which has the lab's"):
        cli.named_file("search")
    with pytest.raises(cli.Fail, match="no agent.py here; run lab in"):
        cli.named_file("agent")
    (tmp_path / "search.py").write_text("")  # not the search folder's
    with pytest.raises(cli.Fail, match="no search/search.py here"):
        cli.named_file("search")
    (tmp_path / "agent.py").write_text("")
    (tmp_path / "search").mkdir()
    (tmp_path / "search" / "search.py").write_text("")
    assert (cli.named_file("search"), cli.named_file("agent"), cli.named_file("x/mine.py")) == (
        "search/search.py", "agent.py", "x/mine.py")
    monkeypatch.chdir(cli.lab.HERE)
    monkeypatch.setattr(cli.labvespa, "FOLDER", pathlib.Path("no-such-folder"))
    with pytest.raises(cli.Fail, match="git checkout -- no-such-folder brings back the lab's"):
        cli.named_file("search")


def test_status_shows_what_is_left_of_the_model_budget():
    assert cli.budget_left({"nick": "maria", "left": {"models": 0.98}}) == " · $0.98 of $1.00 left"
    assert cli.budget_left({}) == "" and cli.budget_left({"left": {"models": None}}) == ""


class Frozen(cli.datetime):
    """10:07 UTC on the first day of 1970, as now."""

    @classmethod
    def now(cls, tz=None):
        return cli.datetime.fromtimestamp(36420.0, tz)


def test_times_are_the_rooms_which_lab_connect_saves(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "datetime", Frozen)
    monkeypatch.setattr(cli.lab, "LAB", tmp_path / "lab.json")
    monkeypatch.setattr(cli.lab, "post", lambda url, body, headers: dict(nick="maria", tz="Europe/Helsinki"))
    assert cli.lab.room_tz() is None  # not connected: this machine's time zone
    assert cli.lab.connect("https://lab.test", "t") == "maria"
    assert cli.when(36420.0) == "12:07" and cli.when(36420.0 + 86400) == "Fri 12:07"  # a GPU box runs on UTC
    cli.lab.save_tz("UTC")
    assert cli.when(36420.0) == "10:07" and json.loads(cli.lab.LAB.read_text())["token"] == "t"


def test_status_shows_each_latest_uploads_grading_on_one_line(monkeypatch):
    monkeypatch.setattr(cli, "datetime", Frozen)
    monkeypatch.setattr(cli.lab, "room_tz", lambda: cli.lab.zoneinfo.ZoneInfo("UTC"))
    search, agent = (show for _, _, show in cli.HANDINS)
    h = dict(at=36420.0, state="graded", preview=0.5978, score=None, error=None)  # 10:07
    plain = lambda text: cli.Text.from_markup(text).plain  # noqa: E731
    assert plain(cli.handin_line(h, search)) == "10:07 · graded · hidden preview 0.598"
    assert plain(cli.handin_line(h | dict(score=0.61), search)) == "10:07 · graded · hidden preview 0.598 · hidden 0.610"
    assert plain(cli.handin_line(h | dict(preview=None), agent)) == "10:07 · graded"
    assert plain(cli.handin_line(h | dict(preview=0.6), agent)) == "10:07 · graded · hidden preview 60%"
    assert plain(cli.handin_line(h | dict(state="failed", error="[app] KeyError: 'x'"), search)) == "10:07 · failed: [app] KeyError: 'x'"
    assert plain(cli.handin_line(h | dict(state="queued", preview=None), agent)) == "10:07 · queued"
    assert plain(cli.handin_line(h | dict(state="grading", preview=None), agent)) == "10:07 · grading"


def test_status_lists_the_latest_uploads_and_why_one_failed(tmp_path, monkeypatch, capsys):
    (tmp_path / "lab.json").write_text(json.dumps(dict(url="https://lab.test", token="t")))
    monkeypatch.setattr(cli.lab, "LAB", tmp_path / "lab.json")
    monkeypatch.setattr(cli, "gpu", lambda: None)
    monkeypatch.setattr(cli.lab, "local_agent_model", lambda: None)
    monkeypatch.setattr(cli.lab, "downloaded", lambda repo, rev: True)
    monkeypatch.setattr(cli, "datetime", Frozen)
    replies = {"/api/me": dict(nick="maria", left=dict(models=0.5), tz="UTC"),
               "/api/handin": dict(handin=dict(at=36420.0, state="failed", preview=None, score=None, error="search.py failed: boom")),
               "/api/agent-handin": dict(handin=dict(at=32100.0, state="queued", preview=None, score=None, error=None))}
    monkeypatch.setattr(cli.lab, "call", lambda path: replies[path])
    cli.cmd_status(None)
    out = capsys.readouterr().out
    assert re.search(r"│ search upload +10:07 · failed: search.py failed: boom +│", out) and re.search(r"│ agent upload +08:55 · queued +│", out)
    replies["/api/agent-handin"] = dict(handin=None)
    cli.cmd_status(None)
    assert re.search(r"│ search upload +10:07", (out := capsys.readouterr().out)) and "agent upload" not in out  # no upload, no row
    assert json.loads(cli.lab.LAB.read_text())["tz"] == "UTC"  # the lab's time zone, saved for the times shown
    replies["/api/handin"]["handin"]["error"] = "agent failed: ModelError: " + "x" * 300
    cli.cmd_status(None)
    out = capsys.readouterr().out.splitlines()
    row = next(i for i, line in enumerate(out) if "search upload" in line)
    assert out[row].rstrip().endswith("… │") and cell_len(out[row]) <= 80 and "xxx" not in out[row + 1]  # one line


def test_eval_search_syncs_vespa_with_its_folder_first(tmp_path, evals, capsys, monkeypatch):
    folder = tmp_path / "search"
    (folder / "app").mkdir(parents=True)
    (folder / "app" / "services.xml").write_text("<services/>")
    (folder / "feed.py").write_text("")
    path = write(folder, "search.py", "def search(ds, question, k=10): raise SystemExit('searched')\n")
    synced = []
    monkeypatch.setattr(cli, "eval_search", lambda *a: synced.append("eval"))

    def refeed(folder, progress, warn):
        raise cli.labvespa.VespaError("fiqa was fed for other document fields: run lab vespa up to feed again")

    monkeypatch.setattr(cli.labvespa, "sync", refeed)
    assert cli.main(["eval", str(path)]) == 1
    assert "run lab vespa up to feed again" in capsys.readouterr().out and synced == []
    monkeypatch.setattr(cli.labvespa, "sync", lambda folder, progress, warn: synced.append(folder) or {})
    assert cli.main(["eval", str(path)]) == 0
    assert synced == [folder, "eval"] and "deployed" in capsys.readouterr().out
    monkeypatch.setattr(cli.labvespa, "sync", lambda folder, progress, warn: synced.append(folder))
    assert cli.main(["eval", str(path)]) == 0
    assert "deployed" not in capsys.readouterr().out
    elsewhere = write(tmp_path, "mine.py", "def search(ds, question, k=10): return []\n")  # not in a search folder
    synced.clear()
    assert cli.main(["eval", str(elsewhere)]) == 0 and synced == ["eval"]


def test_query_search_shows_the_top_k_for_any_question(tmp_path, capsys, monkeypatch):
    path = write(tmp_path, "s.py", """
        def search(ds, question, k=10):
            return [1, 0][:k]
    """)
    ds = cli.lab.Dataset("fiqa", ["a", "b"], ["Tax basics", ""], ["Pay less tax.", "Capital gains explained."], {}, {}, {})
    monkeypatch.setattr(cli.lab, "load", lambda name, split="test": ds)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["query", "search", "how", "to", "lower", "tax"]) == 1  # no ./search here
    monkeypatch.setattr(cli, "named_file", lambda word: str(path))
    assert cli.main(["query", "search", "how", "to", "lower", "tax"]) == 0
    out = capsys.readouterr().out
    assert "2 of 2 documents" in " ".join(out.split()) and out.index("1  Capital gains") < out.index("2  Tax basics · Pay less tax")


def test_query_agent_shows_the_answer_with_its_deals(tmp_path, capsys, monkeypatch):
    path = write(tmp_path, "a.py", """
        def search(query: str) -> list:
            \"\"\"A tool may be named search.\"\"\"
            return []
        def answer(question):
            return {"D0001"}
    """)
    monkeypatch.setattr(cli, "named_file", lambda word: str(path))
    assert cli.main(["query", "agent", "which", "deals", "closed?"]) == 0
    out = capsys.readouterr().out
    assert "answer" in out and "D0001" in out and "Elm Motors" in out and "€56,400" in out
    assert cli.main(["query", "search", "x"]) == 1  # a.py is no search file


def test_search_evals_refuse_the_reranker(tmp_path, capsys, monkeypatch):
    path = write(tmp_path, "s.py", """
        import lab
        def search(ds, question, k=10):
            return lab.rerank_local(ds, question, [0, 1])
    """)
    monkeypatch.setattr(cli.lab, "load", lambda name, split="test": cli.lab.Dataset(name, ["a", "b"], ["", ""], ["", ""], {"q": "x"}, {"q": {"a": 1}}, {}))
    assert cli.main(["eval", str(path)]) == 1
    assert "the reranker is not part of the graded search" in " ".join(capsys.readouterr().out.split())
    assert cli.lab.RERANKER_OFF == ""


def test_lab_page_prints_a_login_link_and_status_says_where_an_upload_waits(connected, capsys, monkeypatch):
    monkeypatch.setattr(cli.lab, "call", lambda path, body: dict(url="https://lab.test/login/ann.1.ab", minutes=10))
    assert cli.main(["page"]) == 0
    assert "open within 10 minutes: https://lab.test/login/ann.1.ab" in capsys.readouterr().out
    line = cli.handin_line(dict(at=0, state="queued", position=3), str)
    assert "queued, 3rd in line" in line and [cli.ordinal(n) for n in (1, 2, 11, 12, 22)] == ["1st", "2nd", "11th", "12th", "22nd"]
