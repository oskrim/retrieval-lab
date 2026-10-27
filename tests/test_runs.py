"""pytest -p no:xdist tests/test_runs.py: saved runs, lab runs, lab compare"""

import gzip
import json
import os

import pytest

import cli
import lab


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "HOME", tmp_path / "home")
    monkeypatch.setattr(cli, "EVALS", tmp_path / "home" / "evals.jsonl")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()  # file names as in a clone: search.py
    return tmp_path


def test_run_names_are_short_stable_and_differ_by_configuration():
    assert cli.run_name("0" * 16, {}) == "able-ant-1"
    assert cli.run_name("3f2a9c0d11e4b7a8", {}) == cli.run_name("3f2a9c0d11e4b7a8", {})
    assert cli.run_name("3f2a9c0d11e4b7a8", {}) == "fancy-badger-40"  # a change to the word lists renames every run
    names = {cli.run_name(f"{i:016x}", {}) for i in range(0, 10**6, 997)}
    assert len(names) > 990 and all(cli.NAME.fullmatch(n) and len(n) <= 20 for n in names)
    assert len(set(cli.ADJECTIVES)) == len(cli.ADJECTIVES) and len(set(cli.NOUNS)) == len(cli.NOUNS)


def test_a_name_another_run_has_counts_on():
    name = cli.run_name("0" * 16, {})
    assert cli.run_name("0" * 16, {name: "0" * 16}) == name  # its own name
    assert cli.run_name("0" * 16, {name: "f" * 16}) == "able-ant-2"


def test_the_run_id_covers_every_file_of_the_search_folder(home):
    (home / "search" / "app").mkdir(parents=True)
    search, schema = home / "search" / "search.py", home / "search" / "app" / "schema.sd"
    search.write_text("def search(ds, question, k=10): ...\n")
    schema.write_text("schema doc {}\n")
    for f in ("feed.py", "analyze.py", "synonyms.json"):
        (home / "search" / f).write_text("")
    plain = home / "search.py"
    plain.write_text(search.read_text())
    first = cli.config_id(search)
    assert cli.config_id(search) == first and cli.config_id(plain) != first
    schema.write_text("schema doc { field x type string {} }\n")
    assert cli.config_id(search) != first
    before = cli.config_id(search)
    for f in ("app/services.xml", "feed.py", "analyze.py", "synonyms.json"):
        (home / "search" / f).write_text("{} # changed")
        assert cli.config_id(search) != before
        before = cli.config_id(search)
    (home / "search" / "app" / ".schema.sd.swp").write_text("editor noise")
    (home / "search" / "__pycache__").mkdir()
    (home / "search" / "__pycache__" / "feed.cpython-312.pyc").write_bytes(b"\0")
    assert cli.config_id(search) == before
    (home / "plain").mkdir()
    (home / "plain" / "search.py").write_text(search.read_text())
    (home / "plain" / "feed.py").write_text("")  # without app/ beside it, not a search folder
    assert cli.config_id(home / "plain" / "search.py") == cli.config_id(plain)


def run(name: str, sets: dict, at: float = 1.0, run_id: str = "a1") -> dict:
    return dict(format=cli.RUN_FORMAT, id=run_id, name=name, at=at, file="/w/search.py", kind="search",
                data=lab.DATA_VERSION, sets=sets)


def test_saving_again_keeps_the_other_datasets_and_only_the_newest_runs(home, monkeypatch):
    fiqa, caiq = dict(score=0.5, questions=[]), dict(score=0.7, questions=[])
    cli.save_run(run("brave-otter-1", {"fiqa dev": fiqa, "caiq test": caiq}))
    cli.save_run(run("brave-otter-1", {"fiqa dev": dict(fiqa, score=0.6)}))
    saved = cli.read_run("brave-otter-1")
    assert saved["sets"]["fiqa dev"]["score"] == 0.6 and saved["sets"]["caiq test"]["score"] == 0.7
    cli.save_run(run("calm-lynx-2", {"fiqa dev": fiqa}, run_id="b2"))
    assert set(cli.read_run("brave-otter-1")["sets"]) == {"fiqa dev", "caiq test"}  # another id: no merge
    monkeypatch.setattr(cli, "KEEP_RUNS", 2)
    os.utime(cli.run_path("brave-otter-1"), (1, 1))
    cli.save_run(run("dapper-mole-3", {"fiqa dev": fiqa}, run_id="c3"))
    assert sorted(p.name for p in cli.run_path("x").parent.iterdir()) == ["calm-lynx-2.json.gz", "dapper-mole-3.json.gz"]
    with pytest.raises(cli.Fail, match="no saved run brave-otter-1"):
        cli.read_run("brave-otter-1")


def test_a_damaged_or_foreign_run_file_is_a_short_message(home):
    cli.run_path("x").parent.mkdir(parents=True)
    cli.run_path("odd-owl-1").write_bytes(b"not gzip")
    with pytest.raises(cli.Fail, match="damaged"):
        cli.read_run("odd-owl-1")
    with gzip.open(cli.run_path("odd-owl-2"), "wt") as f:
        json.dump(dict(format=99, sets={}), f)
    with pytest.raises(cli.Fail, match="another version of lab"):
        cli.read_run("odd-owl-2")


def fake_load(name: str, split: str = "test") -> lab.Dataset:
    """30 documents; question j's relevant documents are j (grade 2) and 10 + j (grade 1)."""
    ids = [f"{name}-d{i:02d}" for i in range(30)]
    qs = {f"{name}-q{j}": f"{name} question {j}: what is the {'tax' if j else 'rent'} rule" for j in range(3)}
    return lab.Dataset(name, ids, [f"title {d}" for d in ids], [f"body of {d}" for d in ids], qs,
                       {q: {ids[j]: 2, ids[10 + j]: 1} for j, q in enumerate(qs)}, {q: set() for q in qs})


FIRST = "def search(ds, question, k=10):\n    return list(range(k))\n"
SECOND = ("def search(ds, question, k=10):  # documents 10 to 12 first\n"
          "    return [10, 11, 12] + [i for i in range(30) if i not in (10, 11, 12)][:k - 3]\n")


@pytest.fixture
def two_runs(home, capsys, monkeypatch):
    monkeypatch.setattr(cli.lab, "load", fake_load)
    search = home / "search" / "search.py"
    search.parent.mkdir()
    names = []
    for code in (FIRST, SECOND):
        search.write_text(code)
        assert cli.main(["eval", "search"]) == 0
        out = capsys.readouterr().out
        names.append(out.split("saved as ")[1].split()[0])
        assert out.rstrip().endswith(f"saved as {names[-1]}") and "next:" not in out
    return names


def test_eval_saves_the_top_20_per_question_under_a_name(two_runs, home):
    first, second = two_runs
    assert first != second
    saved = cli.read_run(first)
    q = saved["sets"]["fiqa dev"]["questions"][0]
    assert q["id"] == "fiqa-q0" and q["score"] == 0.8262 and q["top"] == [f"fiqa-d{i:02d}" for i in range(20)]
    assert q["text"] == "fiqa question 0: what is the rent rule" and set(saved["sets"]) == {"fiqa dev", "caiq test"}
    assert [r["name"] for r in cli.history()] == [first, first, second, second]


def test_the_same_file_again_is_the_same_run(two_runs, home, capsys):
    (home / "search" / "search.py").write_text(FIRST)
    assert cli.main(["eval", "search"]) == 0
    assert f"saved as {two_runs[0]}" in capsys.readouterr().out
    assert [r["name"] for r in cli.saved_runs(cli.history())] == [two_runs[0], two_runs[1]]


def test_runs_lists_the_newest_first_with_the_best_starred(two_runs, capsys):
    first, second = two_runs
    assert cli.main(["runs"]) == 0
    lines = [line.strip() for line in capsys.readouterr().out.splitlines()]
    assert lines[0].split()[:3] == ["run", "fiqa", "dev"]
    assert lines[1].startswith(second) and lines[2].startswith(first)
    assert "★" not in lines[1] and lines[2].count("★") == 2  # the first run scored higher on both
    assert cli.main(["runs", "agent"]) == 1
    assert "no saved runs of agent yet; lab eval agent saves one" in capsys.readouterr().out
    assert cli.main(["runs", "-n", "1"]) == 0
    assert "the newest 1 of 2" in capsys.readouterr().out


def test_compare_shows_each_datasets_change_and_the_questions_that_moved(two_runs, capsys):
    first, second = two_runs
    assert cli.main(["compare"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith(f"{first} ") and f"→ {second}" in out.splitlines()[0]
    fiqa = out.split("fiqa dev")[1].split("caiq test")[0]
    assert "0.587 → 0.519  -0.068   ↑1 ↓2 =0" in fiqa
    moved = [line.split() for line in fiqa.splitlines()[1:] if line.strip()]
    assert [m[:5] for m in moved] == [["fiqa", "3", "0.413", "→", "0.432"], ["fiqa", "1", "0.826", "→", "0.631"],
                                      ["fiqa", "2", "0.521", "→", "0.493"]]  # up first, then the biggest drop first
    assert "fiqa question 2: what is the tax rule" in fiqa
    assert cli.main(["compare", second[:6], first]) == 0  # a name's start is enough
    assert capsys.readouterr().out.splitlines()[0].startswith(f"{second} ")
    assert cli.main(["compare", "nosuch-run-1"]) == 1
    assert "no run nosuch-run-1" in capsys.readouterr().out


def test_compare_needs_two_runs_of_the_file(home, capsys, monkeypatch):
    monkeypatch.setattr(cli.lab, "load", fake_load)
    (home / "search").mkdir()
    (home / "search" / "search.py").write_text(FIRST)
    assert cli.main(["eval", "search"]) == 0
    capsys.readouterr()
    assert cli.main(["compare"]) == 1
    assert "is the only run of" in capsys.readouterr().out


def test_compare_question_shows_where_the_relevant_documents_went(two_runs, capsys):
    first, second = two_runs
    assert cli.main(["compare", "question", "fiqa", "1"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("fiqa 1") and "fiqa question 0: what is the rent rule" in out
    assert " 1 → 4" in out and "11 → 1" in out  # the grade-2 document fell, the grade-1 one rose
    assert "nDCG@10  0.826 → 0.631" in out and first in out and second in out
    side = [line.split("│") for line in out.splitlines() if "│" in line]
    assert side[0][0].split()[:2] == ["1", "✓"] and side[0][0].split()[-1] == "11"  # the newer run first: its 1 was 11th
    assert side[0][1].split()[:2] == ["1", "✓"] and side[0][1].split()[-1] == "4"  # the older run's 1 is 4th now
    header = next(line for line in out.splitlines() if second in line and first in line and "→" not in line
                  and "beside" not in line)
    assert header.index(second) < header.index(first) and " now " not in header and " was " not in header
    assert "not in the top 20" not in out  # every relevant document is in both top 20s
    assert cli.main(["compare", "question", "fiqa", "rent", first, second]) == 0  # words of the question, two runs
    assert capsys.readouterr().out.startswith("fiqa 1")
    assert cli.main(["compare", "question", "2"]) == 1  # question 2 of fiqa or of caiq
    out = capsys.readouterr().out
    assert "fiqa 2" in out and "caiq 2" in out and "2 questions match 2" in out


def test_compare_question_leaves_out_relevant_documents_in_neither_top_20(home, capsys, monkeypatch):
    monkeypatch.setattr(cli.lab, "load", fake_load)
    search = home / "search" / "search.py"
    search.parent.mkdir()
    for order in ("range(29, 9, -1)", "range(10, 30)"):  # neither run finds documents 0 to 9
        search.write_text(f"def search(ds, question, k=10):\n    return list({order})[:k]\n")
        assert cli.main(["eval", "search", "--data", "fiqa"]) == 0
    capsys.readouterr()
    assert cli.main(["compare", "question", "fiqa", "1"]) == 0
    lines = [" ".join(line.split()) for line in capsys.readouterr().out.splitlines()]
    assert "1 relevant in neither top 20" in lines and not any("– → –" in line for line in lines)


def test_worst_questions_are_numbered_as_lab_compare_question_numbers_them(home, capsys, monkeypatch):
    monkeypatch.setattr(cli.lab, "load", fake_load)
    search = home / "search" / "search.py"
    search.parent.mkdir()
    search.write_text(SECOND)
    assert cli.main(["eval", "search", "--data", "fiqa", "--worst", "1"]) == 0
    assert "fiqa 3 · id fiqa-q2" in capsys.readouterr().out  # its third question scores lowest


def test_compare_an_agent_question(home, capsys, monkeypatch):
    monkeypatch.setattr(cli, "require_agent", lambda: ("http://localhost:8000/v1/chat/completions", {}, "fake-model"))
    monkeypatch.setattr(cli.lab, "crm_dev", lambda: [("first question", {"D0001"}), ("second question", {"D0002"})])
    for found in ("{'D0001'}", "{'D0002'}  # a size of its own: Python checks the cached bytecode by size and second"):
        (home / "agent.py").write_text(f"def answer(question):\n    return {found}\n")
        assert cli.main(["eval", "agent"]) == 0
    capsys.readouterr()
    assert cli.main(["compare", "agent"]) == 0
    assert "↑1 ↓1 =0" in capsys.readouterr().out
    assert cli.main(["compare", "question", "crm", "2"]) == 0
    out = capsys.readouterr().out
    lines = [line.split() for line in out.splitlines()]
    assert "second question" in out and ["expected", "D0002"] in lines
    assert "✗ found D0001 · missing D0002 · extra D0001" in out and "✓ found D0002 · missing – · extra –" in out


AGENT = '''import lab

SYSTEM = "{system}"


def find(query="", filters=None, limit=None):
    return dict(total=1, deals=[dict(deal_id="D0001", company="Acme Oy")])


def answer(question):
    return lab.crm_agent(question, [lab.ATTR_SEARCH], dict(search=find), system=SYSTEM)["deals"]
'''


def scripted_model(url, body, headers, timeout=120):
    """The agent model: one search (plain text with the "plain" system prompt, a filter with "filters"), then an
    answer: D0001, and D0002 too with "filters"."""
    messages = body["messages"]
    filters = messages[0]["content"] == "filters"
    usage = dict(total_tokens=1500, usd=0.0015)
    if messages[-1]["role"] == "user":
        args = dict(filters=dict(industry="Pharmaceuticals")) if filters else dict(query="pharma", filters=None)
        call = dict(id="c1", type="function", function=dict(name="search", arguments=json.dumps(args)))
        return dict(choices=[dict(message=dict(role="assistant", content=None, tool_calls=[call]))], usage=usage)
    deals = "D0001, D0002" if filters else "D0001"
    return dict(choices=[dict(message=dict(role="assistant", content=f"Found them.\nDEALS: {deals}"))], usage=usage)


@pytest.fixture
def agent_runs(home, capsys, monkeypatch):
    monkeypatch.setattr(cli, "require_agent", lambda: ("http://localhost:8000/v1/chat/completions", {}, "fake-model"))
    monkeypatch.setattr(cli.lab, "agent_endpoint", lambda: ("http://localhost:8000/v1/chat/completions", {}, "fake-model"))
    monkeypatch.setattr(cli.lab, "post", scripted_model)
    monkeypatch.setattr(cli.lab, "crm_dev", lambda: [("Pharma deals?", {"D0001", "D0002"}), ("Acme's deals?", {"D0001"})])
    names = []
    for system in ("plain", "filters"):
        (home / "agent.py").write_text(AGENT.format(system=system))
        assert cli.main(["eval", "agent"]) == 0
        names.append(capsys.readouterr().out.split("saved as ")[1].split()[0])
    return names


def lines_of(out: str) -> list[str]:
    assert max(map(len, out.splitlines())) <= 80
    return [" ".join(line.split()) for line in out.splitlines()]


def test_trace_shows_one_questions_tool_calls_results_and_answer(agent_runs, capsys):
    first, second = agent_runs
    assert cli.main(["trace", "crm", "1"]) == 0  # the newest agent run
    lines = lines_of(capsys.readouterr().out)
    assert lines[0] == f"crm 1 {second} · ✓ found 2 of 2 · 2 turns · 3,000 tokens · $0.003"
    assert lines[1] == "Pharma deals?"
    assert lines[2] == "1 search filters={industry: Pharmaceuticals} → 1 deal: D0001 Acme Oy"
    assert lines[3:6] == ["2 answer D0001 D0002", "says Found them. DEALS: D0001, D0002", "expected D0001 D0002"]
    assert cli.main(["trace", "1", first[:5]]) == 0  # the start of a run's name
    lines = lines_of(capsys.readouterr().out)
    assert lines[0] == f"crm 1 {first} · ✗ found 1 of 2 · 2 turns · 3,000 tokens · $0.003"
    assert lines[2] == '1 search query="pharma" → 1 deal: D0001 Acme Oy'
    assert lines[-1] == "missing D0002 extra –"
    assert cli.main(["trace", "crm", "acme", "--json"]) == 0
    saved = json.loads(capsys.readouterr().out)
    assert saved["run"] == second and saved["question"] == 2 and saved["top"] == ["D0001", "D0002"]
    assert saved["trace"][0]["calls"][0]["result"] == "1 deal: D0001 Acme Oy" and len(gzip.compress(
        json.dumps(saved).encode())) < 1000


def test_compare_shows_turns_and_tokens_and_both_trajectories(agent_runs, capsys):
    first, second = agent_runs
    assert cli.main(["compare", "agent"]) == 0
    rows = [line for line in lines_of(capsys.readouterr().out) if line.startswith("crm ")]
    assert rows[1].startswith("crm 1 0.000 → 1.000 +1.000 2→2 turns 3.0k→3.0k tok Pharma deals?")
    assert cli.main(["compare", "question", "crm", "1"]) == 0
    out = capsys.readouterr().out
    blocks = out.split(f"\n  {first} · ")[1].split(f"\n  {second} · ")
    assert blocks[0].startswith("✗ found 1 of 2") and blocks[1].startswith("✓ found 2 of 2")
    marked = [" ".join(line.split()) for line in out.splitlines() if line.startswith("›")]
    assert marked == ['› 1 search query="pharma" → 1 deal: D0001 Acme Oy',
                      "› 1 search filters={industry: Pharmaceuticals} → 1 deal: D0001 Acme Oy"]


def test_trace_of_a_run_saved_before_trajectories_says_so(agent_runs, capsys):
    saved = cli.read_run(agent_runs[1])
    for q in saved["sets"]["crm dev"]["questions"]:
        del q["trace"]
    cli.save_run(saved)
    assert cli.main(["trace", "crm", "1"]) == 1
    assert "saved before lab eval recorded trajectories" in capsys.readouterr().out


def test_each_eval_of_an_unchanged_agent_is_kept_under_its_own_number(agent_runs, home, capsys):
    first, second = agent_runs
    assert cli.main(["eval", "agent"]) == 0  # agent.py as the second eval had it: the model may answer otherwise
    third = capsys.readouterr().out.split("saved as ")[1].split()[0]
    assert third == f"{second}#2" and cli.read_run(second)["name"] == second and cli.read_run(third)["name"] == third
    assert cli.main(["runs", "agent"]) == 0
    listed = [line.split()[0] for line in capsys.readouterr().out.splitlines()[1:] if line.strip() and "★ the" not in line]
    assert listed == [third, second, first]
    assert cli.main(["trace", "crm", "1", third]) == 0 and capsys.readouterr().out.startswith(f"crm 1  {third}")
    assert cli.main(["trace", "crm", "1", second]) == 0 and capsys.readouterr().out.startswith(f"crm 1  {third}")  # the latest
    assert cli.main(["trace", "crm", "1", f"{second}#1"]) == 0 and capsys.readouterr().out.startswith(f"crm 1  {second} ")
    assert cli.main(["compare", "agent"]) == 0
    assert f"{second} " in (out := capsys.readouterr().out) and f"→ {third}" in out  # the two newest evals of the file
