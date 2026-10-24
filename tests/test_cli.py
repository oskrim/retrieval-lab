"""pytest -p no:xdist tests"""

import pathlib
import sys
import textwrap
import threading

import numpy as np
import pytest

import cli


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
    monkeypatch.setattr(cli, "require_agent", lambda: None)
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


def test_eval_history_skips_a_line_cut_short(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "EVALS", tmp_path / "evals.jsonl")
    cli.remember(tmp_path / "a.py", "search", "fiqa dev", 0.5, 0.4, 0.6, 500)
    with open(cli.EVALS, "a") as f:
        f.write('{"at": 1, "file": "b.py", "kind": "sea')
    assert cli.best_before("search", "fiqa dev")["score"] == 0.5


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
    monkeypatch.setattr(cli, "require_agent", lambda: None)
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
