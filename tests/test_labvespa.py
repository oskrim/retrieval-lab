"""pytest -p no:xdist tests"""

import io
import json
import math
import pathlib
import zipfile

import numpy as np
import pytest

import labvespa

SD = """
# a comment with field fake type string {
schema fiqa {
  document fiqa {
    field id type string {
      indexing: attribute | summary
    }
    field dense_float type tensor<bfloat16>(x[4]) {
      indexing: attribute
      attribute {
        distance-metric: dotproduct
      }
    }
  }
  field body_pairs type string {
    indexing: input body | index
  }
  rank-profile hybrid {
    first-phase {
      expression: 1
    }
  }
}
"""


def test_fields_are_the_document_blocks_only():
    assert labvespa.fields(SD, "fiqa") == {"id": "string", "dense_float": "tensor<bfloat16>(x[4])"}
    with pytest.raises(labvespa.VespaError, match="no 'document caiq"):
        labvespa.fields(SD, "caiq")


def test_the_feed_fingerprint_moves_with_document_fields_and_the_feeds_code_only():
    before = labvespa.fingerprint(SD, "fiqa", "code")
    assert labvespa.fingerprint(SD.replace("expression: 1", "expression: 2").replace("# a comment", "# other"), "fiqa",
                                "code") == before
    assert labvespa.fingerprint(SD.replace("bfloat16", "float"), "fiqa", "code") != before
    assert labvespa.fingerprint(SD, "fiqa", "other code") != before


def test_the_feed_digest_covers_feed_py_and_the_folders_modules_it_imports(tmp_path):
    (tmp_path / "feed.py").write_text("import json\nimport analyze\nfrom helpers.text import clean\n")
    (tmp_path / "analyze.py").write_text("import words\n")
    (tmp_path / "words.py").write_text("STOP = 1\n")
    (tmp_path / "helpers").mkdir()
    (tmp_path / "search.py").write_text("import analyze\n")
    before = labvespa.feed_digest(tmp_path)
    (tmp_path / "search.py").write_text("import analyze  # tuned\n")
    assert labvespa.feed_digest(tmp_path) == before  # the query is not part of the feed
    for f in ("words.py", "analyze.py", "feed.py"):
        (tmp_path / f).write_text((tmp_path / f).read_text() + "# changed\n")
        assert labvespa.feed_digest(tmp_path) != before
        before = labvespa.feed_digest(tmp_path)


def test_hexed_writes_each_cell_type_big_endian():
    x = np.array([1.0, -2.0], dtype=np.float32)
    assert labvespa.hexed(x, "float") == "3f800000c0000000"
    assert labvespa.hexed(x, "bfloat16") == "3f80c000"
    assert labvespa.hexed(x, "double") == "3ff0000000000000c000000000000000"
    assert labvespa.hexed(np.array([-1, 127]), "int8") == "ff7f"


TEMPLATE = SD.replace("schema fiqa {", "schema dataset {").replace("document fiqa {", "document dataset {")


def app(tmp_path, sd: str = TEMPLATE):
    (tmp_path / "services.xml").write_text("<services/>")
    (tmp_path / "schema.sd").write_text(sd)
    return tmp_path


def folder(tmp_path, feed: str = "def documents(ds, fields):\n    yield from ()\n"):
    """A search folder: app/ with TEMPLATE as its schema, and feed.py."""
    (tmp_path / "app").mkdir(parents=True)
    app(tmp_path / "app")
    (tmp_path / "feed.py").write_text(feed)
    return tmp_path


def test_a_search_folder_holds_app_and_feed_py(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert labvespa.search_folder() == labvespa.LAB_FOLDER  # no ./search: the lab's own
    path = folder(tmp_path / "search")
    assert labvespa.search_folder() == path and labvespa.search_folder("search") == path
    with pytest.raises(labvespa.VespaError, match="search/app is an application; give the search folder"):
        labvespa.search_folder("search/app")
    (path / "feed.py").unlink()
    with pytest.raises(labvespa.VespaError, match="search has no feed.py"):
        labvespa.search_folder()
    with pytest.raises(labvespa.VespaError, match="elsewhere is not a search folder"):
        labvespa.search_folder("elsewhere")


def test_each_datasets_schema_is_schema_sd_with_only_the_schema_and_document_named_after_it(tmp_path):
    path = app(tmp_path, TEMPLATE.replace("# a comment", "# schema dataset {, document dataset {"))
    for name in ("fiqa", "caiq"):
        sd = labvespa.schema(path, name)
        assert f"schema {name} {{" in sd and f"  document {name} {{" in sd and "# schema dataset {, document dataset {" in sd
        assert labvespa.fields(sd, name) == labvespa.fields(SD, "fiqa")
    with pytest.raises(labvespa.VespaError, match="one with 'document dataset'"):
        labvespa.schema(app(tmp_path, TEMPLATE.replace("document dataset {", "document other {")), "caiq")


def test_the_package_makes_every_datasets_schema_and_says_a_written_one_is_not_used(tmp_path):
    path = app(tmp_path)
    (path / "schemas").mkdir()
    (path / "schemas" / "caiq.sd").write_text("schema caiq { stale }")
    warned = []
    with zipfile.ZipFile(io.BytesIO(labvespa.package(path, warned.append))) as z:
        assert sorted(z.namelist()) == ["schemas/caiq.sd", "schemas/fiqa.sd", "services.xml"]
        assert z.read("schemas/caiq.sd").decode() == labvespa.schema(path, "caiq")
        assert z.read("schemas/fiqa.sd").decode() == labvespa.schema(path, "fiqa")
    assert len(warned) == 1 and "caiq.sd is not used" in warned[0] and "edit schema.sd only" in warned[0]


def test_the_lab_app_has_one_schema_for_both_datasets():
    app = labvespa.LAB_FOLDER / "app"
    assert not (app / "schemas").exists() and (app / "schema.sd").is_file()
    assert labvespa.fields(labvespa.schema(app, "caiq"), "caiq")["dense_bits"] == "tensor<int8>(x[128])"


def test_query_errors_put_the_cause_first_and_cut_tensors():
    vector = "[" + ", ".join(["0.0123456789"] * 1024) + "]"
    msg = labvespa.brief_query_error({"summary": "Illegal query", "message": f"Could not set 'ranking.features.query(qc)' "
                                      f"to '{vector}': No profile named 'hybrid-cos' exists in schemas [caiq]"})
    lines = msg.splitlines()
    assert lines[0] == "Vespa: No profile named 'hybrid-cos' exists in schemas [caiq]"
    assert len(lines) == 2 and len(msg) < 400 and "0.0123456789, 0.0123456789, 0.0123456789" not in msg


def test_deploy_errors_put_the_failing_feature_first_without_the_passing_profiles():
    message = "\n".join(["Invalid application: Error in rank setup in schema 'fiqa' for content cluster 'content'. Details:",
                         *[f"INFO: rank profile 'p{i}': pass" for i in range(12)],
                         "WARNING: invalid rank feature cosine(dense_float): unknown basename: 'cosine'",
                         "          ... needed by function firstphase",
                         "ERROR: rank profile 'hybrid-cos': FAIL"] * 2)
    lines = labvespa.brief_deploy_error(message).splitlines()
    assert lines[0] == "invalid rank feature cosine(dense_float): unknown basename: 'cosine'"
    assert len(lines) == 4 and not any("pass" in line for line in lines)


def test_a_search_on_a_stale_feed_fails(tmp_path, monkeypatch):
    path = folder(tmp_path / "mine")
    fed = {"fiqa": labvespa.fingerprint(SD, "fiqa", labvespa.feed_digest(path))}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(labvespa, "FOLDER", pathlib.Path("mine"))
    monkeypatch.setattr(labvespa, "_FED_CHECKED", set())
    monkeypatch.setattr(labvespa, "deployed", lambda name: SD.replace("fiqa", name))
    monkeypatch.setattr(labvespa, "fed", lambda: fed)
    monkeypatch.setattr(labvespa, "state", lambda: "running")
    assert labvespa.unfed() == {"caiq": "not fed, or its last feed did not finish"}
    (path / "feed.py").write_text((path / "feed.py").read_text() + "# changed\n")
    stale = "fed for other document fields than the deployed schema's, or by another mine/feed.py"
    assert labvespa.unfed() == {"fiqa": stale, "caiq": "not fed, or its last feed did not finish"}
    fed["fiqa"] = labvespa.fingerprint(SD.replace("bfloat16", "float"), "fiqa", labvespa.feed_digest(path))
    assert labvespa.unfed()["fiqa"] == stale
    monkeypatch.setattr(labvespa, "http", lambda *a, **k: pytest.fail("no query on a stale feed"))
    with pytest.raises(labvespa.VespaError, match="^fiqa is fed for other document fields .*run lab vespa up"):
        labvespa.query("fiqa", "select * from fiqa where true", "hybrid")


def fed_lines(path, ds, wanted, warned=None):
    return [json.loads(line) for line in labvespa.lines(ds.name, path, ds, wanted, (warned if warned is not None else []).append)]


def test_the_template_feed_writes_the_declared_fields_and_the_transport_warns_of_the_rest(monkeypatch):
    ds = labvespa.lab.Dataset("caiq", ["a", "b"], ["T\x10", ""], ["x", "y"], {}, {}, {})
    monkeypatch.setattr(labvespa.lab, "vectors", lambda ds: (np.array([[0.1, -0.2], [0.0, 0.25]], dtype=np.float32), None))
    warned = []
    docs = fed_lines(labvespa.LAB_FOLDER, ds, {"id": "string", "title": "string", "dense_float": "tensor<float>(x[2])",
                                               "mine": "string"}, warned)
    assert warned == ["schema.sd field mine: feed.py writes no value into it, so it stays empty (caiq)"]
    assert docs[0] == {"put": "id:caiq:caiq::a", "fields": {"id": "a", "title": "T ", "dense_float": {"values": "3dcccccdbe4ccccd"}}}
    assert len(docs) == 2
    with pytest.raises(labvespa.VespaError, match=r"ValueError: schema.sd field dense_bits needs int8 cells.*\(feed.py:\d+\)"):
        fed_lines(labvespa.LAB_FOLDER, ds, {"dense_bits": "tensor<float>(x[2])"})


def test_the_template_chunks_keep_the_posts_id_and_vectors(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("template_feed", labvespa.LAB_FOLDER / "feed.py")
    feed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feed)
    monkeypatch.setattr(feed, "CHUNK_WORDS", 2)
    monkeypatch.setattr(labvespa.lab, "vectors", lambda ds: (np.array([[0.1, -0.2], [0.0, 0.25]], dtype=np.float32), None))
    ds = labvespa.lab.Dataset("caiq", ["a", "b"], ["T", ""], ["one two three", ""], {}, {}, {})
    got = list(feed.chunks(ds, {"id": "string", "body": "string", "dense_float": "tensor<float>(x[2])"}))
    assert got == [("a#0", {"id": "a", "body": "one two", "dense_float": {"values": "3dcccccdbe4ccccd"}}),
                   ("a#1", {"id": "a", "body": "three", "dense_float": {"values": "3dcccccdbe4ccccd"}}),
                   ("b#0", {"id": "b", "body": "", "dense_float": {"values": "000000003e800000"}})]


@pytest.mark.parametrize("feed, error", [
    ("def documents(ds, fields):\n    yield {'id': 'a'}\n", r"yields \(document id, fields\) pairs; it yielded \{'id': 'a'\}"),
    ("def documents(ds, fields):\n    yield 'a', {'id': 'a', 'tags': 'x'}\n", "writes tags into document a, which schema.sd"),
    ("def documents(ds, fields):\n    yield 'a', {'id': object()}\n", "document a is not JSON"),
    ("def documents(ds, fields):\n    return [1 / 0]\n", r"feed.py raised ZeroDivisionError: division by zero \(feed.py:2\)"),
    ("import nothing_here\n", r"feed.py raised ModuleNotFoundError: No module named 'nothing_here' \(feed.py:1\)"),
    ("def documents(ds, fields)\n", r"feed.py raised SyntaxError: .* \(feed.py:1\)"),
    ("x = 1\n", "defines no documents"),
])
def test_a_broken_feed_py_says_what_is_wrong_with_its_line(tmp_path, feed, error):
    ds = labvespa.lab.Dataset("caiq", ["a"], [""], [""], {}, {}, {})
    with pytest.raises(labvespa.VespaError, match=error):
        fed_lines(folder(tmp_path, feed), ds, {"id": "string"})


@pytest.mark.parametrize("stderr, hint", [
    ("docker: Error response from daemon: Bind for 127.0.0.1:18080 failed: port is already allocated\n\n"
     "Run 'docker run --help' for more information\n", "Bind for 127.0.0.1:18080 failed: port is already allocated; Vespa needs ports"),
    ("permission denied while trying to connect to the Docker daemon socket", "sudo usermod -aG docker"),
    ("Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?",
     "sudo systemctl start docker"),
])
def test_docker_failures_say_what_to_do(stderr, hint, monkeypatch):
    monkeypatch.setattr(labvespa.subprocess, "run", lambda *a, **k: labvespa.subprocess.CompletedProcess(a, 125, "", stderr))
    with pytest.raises(labvespa.VespaError, match="^docker run failed: ") as e:
        labvespa.docker("run", "x")
    assert hint in str(e.value) and "--help" not in str(e.value)


def test_start_replaces_a_container_that_never_started(monkeypatch):
    calls, states = [], iter(["created", "running"])
    monkeypatch.setattr(labvespa, "state", lambda: next(states))
    monkeypatch.setattr(labvespa, "docker", lambda *a, **k: calls.append(a[0]) or "")
    monkeypatch.setattr(labvespa, "reachable", lambda url: {"status": {"code": "up"}})
    labvespa.start()
    assert calls == ["rm", "run"]


def test_a_container_that_stops_while_starting_fails_at_once(monkeypatch):
    monkeypatch.setattr(labvespa, "state", lambda: "exited")
    logs = "info: starting\npanic: this host [localhost] should not run a config server\n"
    monkeypatch.setattr(labvespa.subprocess, "run", lambda *a, **k: labvespa.subprocess.CompletedProcess(a, 0, logs, ""))
    with pytest.raises(labvespa.VespaError, match="stopped while starting: panic: this host"):
        labvespa.alive()


def test_a_deploy_that_empties_a_field_unmarks_its_feed(tmp_path, monkeypatch):
    marks = {"fiqa": "f1", "caiq": "c1"}
    monkeypatch.setattr(labvespa, "state", lambda: "running")
    monkeypatch.setattr(labvespa, "fed", lambda: dict(marks))
    monkeypatch.setattr(labvespa, "mark", lambda done: (marks.clear(), marks.update(done)))
    monkeypatch.setattr(labvespa, "wait", lambda *a, **k: None)
    answer = {"configChangeActions": {"refeed": [{"documentType": "fiqa", "messages": ["field type changed"]}]}}
    monkeypatch.setattr(labvespa, "http", lambda *a, **k: (200, answer))
    monkeypatch.setattr(labvespa, "docker", lambda *a, **k: "")
    labvespa.deploy(folder(tmp_path), warn=lambda m: None)
    assert marks == {"caiq": "c1"}


def test_a_blocked_feed_says_why_in_one_line(tmp_path, monkeypatch):
    error = ('feed: got status 507 ({"message":"[UNKNOWN(251009) @ tcp/lab-vespa:19115/default]: ReturnCode(NO_SPACE, '
             "External feed is blocked due to resource exhaustion: in content cluster 'content': disk on node 0 [lab-vespa] "
             'is 80.2% full (the configured limit is 80.0%). See https://docs.vespa.ai/en/writing/feed-block.html) "})')

    class Feed:  # vespa feed in the container: reads every line, then fails
        def __init__(self, cmd, stdin, stdout, stderr, text):
            self.stdin, self.stderr = open(tmp_path / "in", "w"), stderr
            stderr.write(error + "\n")

        def wait(self, timeout):
            return 1

        def poll(self):
            return 1

    monkeypatch.setattr(labvespa.subprocess, "Popen", Feed)
    monkeypatch.setattr(labvespa.lab, "load", lambda name: None)
    monkeypatch.setattr(labvespa, "lines", lambda name, folder, ds, wanted, warn: iter(["{}\n"]))
    monkeypatch.setattr(labvespa, "state", lambda: "running")
    with pytest.raises(labvespa.VespaError) as e:
        labvespa.feed("fiqa", folder(tmp_path))
    assert str(e.value) == ("Vespa blocks the feed: disk on node 0 [lab-vespa] is 80.2% full (the configured limit is "
                            "80.0%); free some space, then lab vespa up")


def test_limits_fit_a_small_box(monkeypatch):
    monkeypatch.setattr(labvespa.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(labvespa.os, "sysconf", lambda name: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 16 * 2**30 // 4096}[name])
    assert labvespa.limits() == ("--memory", "12g", "--cpus", "4")


def vespa_search():
    import importlib.util
    import sys

    sys.path.insert(0, str(labvespa.LAB_FOLDER))  # its search.py imports analyze.py beside it
    spec = importlib.util.spec_from_file_location("vespa_search", labvespa.LAB_FOLDER / "search.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_vespa_query_matches_any_word_and_sends_each_synonym_phrase_as_one_equiv_term():
    search = vespa_search()
    question = "For Octopus Deploy: is data encrypted?"
    terms, pairs = search.words(question)
    assert terms == ["data", "encrypted"] and ["octopus", "deploy"] in pairs
    ds = labvespa.lab.Dataset("caiq", ["a", "b"], ["", ""], ["Octopus Deploy", "data"], {}, {}, {})
    yql = search.yql(ds, question, terms, pairs)
    weight = 2 * math.log(1 + 1.5 / 1.5)  # lab's idf of "octopus" and of "deploy": one of two documents has each
    assert ('(default contains "data" or default contains "encrypted" or default contains '
            f'({{significance: {weight:.4f}}}equiv(phrase("octopus", "deploy"), "octopus", "octopusdeploy")))') in yql
    assert "weakAnd" not in yql and "{targetHits:100}nearestNeighbor(dense_bits, qbits)" in yql


def test_the_package_takes_files_dated_before_1980(tmp_path):
    import os

    path = app(tmp_path)
    for f in (path / "services.xml", path / "schema.sd"):  # a hand-in unpacked by tar keeps the pack's mtime 0
        os.utime(f, (0, 0))
    with zipfile.ZipFile(io.BytesIO(labvespa.package(path, print))) as z:
        assert sorted(z.namelist()) == ["schemas/caiq.sd", "schemas/fiqa.sd", "services.xml"]


def test_a_feed_into_a_vespa_with_documents_starts_from_an_empty_one(tmp_path, monkeypatch):
    path, calls = folder(tmp_path), []
    keys = {name: labvespa.fingerprint(labvespa.schema(path / "app", name), name, labvespa.feed_digest(path))
            for name in labvespa.DATASETS}
    marker = {"fiqa": keys["fiqa"], "caiq": "an older feed"}
    monkeypatch.setattr(labvespa.lab, "DATA", tmp_path)
    (tmp_path / labvespa.lab.DATA_VERSION).mkdir()
    monkeypatch.setattr(labvespa, "start", lambda progress: calls.append("start"))
    monkeypatch.setattr(labvespa, "deploy", lambda folder, progress, warn: calls.append("deploy") or {})
    monkeypatch.setattr(labvespa, "fed", lambda: dict(marker))
    monkeypatch.setattr(labvespa, "mark", lambda done: (marker.clear(), marker.update(done)))
    monkeypatch.setattr(labvespa, "holds_documents", lambda: "down" not in calls)
    monkeypatch.setattr(labvespa, "down", lambda: calls.append("down") or marker.clear())
    monkeypatch.setattr(labvespa, "feed", lambda name, folder, advance, warn: calls.append(f"feed {name}") or 1.0)
    assert labvespa.up(path)["fed"] == {"fiqa": 1.0, "caiq": 1.0}
    assert calls == ["start", "deploy", "down", "start", "deploy", "feed fiqa", "feed caiq"] and marker == keys
    calls.clear()
    labvespa.up(path)  # fed for this folder: nothing to do
    assert calls == ["start", "deploy"]


def test_the_template_search_keeps_each_posts_best_chunk():
    search = vespa_search()
    hits = [("p1", 9.0), ("p2", 8.0), ("p1", 7.0), ("p3", 6.0), ("p2", 5.0)]
    assert search.best_chunks(hits) == [("p1", 9.0), ("p2", 8.0), ("p3", 6.0)]


def test_sync_deploys_a_changed_app_and_sends_a_changed_document_to_lab_vespa_up(tmp_path, monkeypatch):
    path, calls = folder(tmp_path), []
    keys = {name: labvespa.fingerprint(labvespa.schema(path / "app", name), name, labvespa.feed_digest(path))
            for name in labvespa.DATASETS}
    container = {labvespa.APP_MARKER: ""}

    def docker(*args, stdin=None, **k):  # cat > APP_MARKER writes it, cat APP_MARKER reads it
        if "-i" in args:
            container[labvespa.APP_MARKER] = stdin
        return container[labvespa.APP_MARKER]

    monkeypatch.setattr(labvespa, "state", lambda: "running")
    monkeypatch.setattr(labvespa, "fed", lambda: dict(keys))
    monkeypatch.setattr(labvespa, "docker", docker)
    monkeypatch.setattr(labvespa, "wait", lambda *a, **k: None)
    monkeypatch.setattr(labvespa, "http", lambda *a, **k: calls.append("deploy") or (200, {}))
    assert labvespa.sync(path) == {} and calls == ["deploy"]  # nothing deployed yet
    assert labvespa.sync(path) is None and calls == ["deploy"]
    sd = path / "app" / labvespa.TEMPLATE
    sd.write_text(sd.read_text().replace("expression: 1", "expression: 2"))  # a rank profile: deployed, not fed
    assert labvespa.sync(path) == {} and calls == ["deploy"] * 2
    assert labvespa.sync(path) is None and calls == ["deploy"] * 2
    sd.write_text(sd.read_text().replace("x[4]", "x[8]"))  # a document field
    with pytest.raises(labvespa.VespaError, match=r"^fiqa and caiq were fed for other document fields in .*schema.sd or "
                       r"another .*feed.py: run lab vespa up to feed again \(about 2 minutes\)$"):
        labvespa.sync(path)
    monkeypatch.setattr(labvespa, "fed", lambda: {})
    with pytest.raises(labvespa.VespaError, match="^fiqa and caiq are not fed: run lab vespa up"):
        labvespa.sync(path)
    monkeypatch.setattr(labvespa, "state", lambda: None)
    with pytest.raises(labvespa.VespaError, match="Vespa is not running; lab vespa up starts it"):
        labvespa.sync(path)
    assert calls == ["deploy"] * 2


def test_the_search_folder_starts_with_the_lab_synonyms():
    assert json.loads((labvespa.LAB_FOLDER / "synonyms.json").read_text())["groups"] == labvespa.lab.synonyms()


def test_the_template_search_sends_late_tokens_only_to_a_profile_that_uses_them(tmp_path, monkeypatch):
    search = vespa_search()
    assert search.uses_late("hybrid") is False  # its late phases wait commented out
    schema = tmp_path / "app" / "schema.sd"
    schema.parent.mkdir()
    schema.write_text("rank-profile a inherits common {\n  # late_int8 in a comment\n}\n"
                      "rank-profile b inherits common {\n  second-phase { expression: late_int8 }\n}\n")
    monkeypatch.setattr(search, "__file__", str(tmp_path / "search.py"))
    search._LATE.clear()
    assert search.uses_late("a") is False and search.uses_late("b") is True
