"""pytest -p no:xdist tests"""

import pathlib
import textwrap

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
