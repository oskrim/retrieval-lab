"""pytest -p no:xdist tests"""

import io
import json
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


def test_the_feed_fingerprint_moves_with_document_fields_only():
    before = labvespa.fingerprint(SD, "fiqa")
    assert labvespa.fingerprint(SD.replace("expression: 1", "expression: 2").replace("# a comment", "# other"), "fiqa") == before
    assert labvespa.fingerprint(SD.replace("bfloat16", "float"), "fiqa") != before


def test_hexed_writes_each_cell_type_big_endian():
    x = np.array([1.0, -2.0], dtype=np.float32)
    assert labvespa.hexed(x, "float") == "3f800000c0000000"
    assert labvespa.hexed(x, "bfloat16") == "3f80c000"
    assert labvespa.hexed(x, "double") == "3ff0000000000000c000000000000000"
    assert labvespa.hexed(np.array([-1, 127]), "int8") == "ff7f"


def app(tmp_path, sd: str = SD):
    (tmp_path / "schemas").mkdir(parents=True, exist_ok=True)
    (tmp_path / "services.xml").write_text("<services/>")
    (tmp_path / "schemas" / "fiqa.sd").write_text(sd)
    return tmp_path


def test_caiq_sd_is_fiqa_sd_with_only_the_schema_and_document_renamed(tmp_path):
    caiq = labvespa.schema(app(tmp_path, SD.replace("# a comment", "# schema fiqa {, document fiqa {")), "caiq")
    assert "schema caiq {" in caiq and "  document caiq {" in caiq and "# schema fiqa {, document fiqa {" in caiq
    assert labvespa.fields(caiq, "caiq") == labvespa.fields(SD, "fiqa")
    with pytest.raises(labvespa.VespaError, match="one with 'document fiqa'"):
        labvespa.schema(app(tmp_path, SD.replace("document fiqa {", "document other {")), "caiq")


def test_the_package_makes_caiq_sd_and_says_a_written_one_is_not_used(tmp_path):
    path = app(tmp_path)
    (path / "schemas" / "caiq.sd").write_text("schema caiq { stale }")
    warned = []
    with zipfile.ZipFile(io.BytesIO(labvespa.package(path, warned.append))) as z:
        assert sorted(z.namelist()) == ["schemas/caiq.sd", "schemas/fiqa.sd", "services.xml"]
        assert z.read("schemas/caiq.sd").decode() == labvespa.schema(path, "caiq")
    assert len(warned) == 1 and "caiq.sd is not used" in warned[0] and "edit fiqa.sd only" in warned[0]


def test_the_lab_app_has_one_schema_for_both_datasets():
    assert sorted(p.name for p in (labvespa.APP / "schemas").iterdir()) == ["fiqa.sd"]
    assert labvespa.fields(labvespa.schema(labvespa.APP, "caiq"), "caiq")["dense_bits"] == "tensor<int8>(x[128])"


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


def test_a_search_on_a_feed_that_does_not_match_the_deployed_schema_fails(monkeypatch):
    monkeypatch.setattr(labvespa, "_FED_CHECKED", set())
    monkeypatch.setattr(labvespa, "deployed", lambda name: SD.replace("fiqa", name))
    monkeypatch.setattr(labvespa, "fed", lambda: {"fiqa": labvespa.fingerprint(SD.replace("bfloat16", "float"), "fiqa")})
    monkeypatch.setattr(labvespa, "state", lambda: "running")
    assert labvespa.unfed() == {"fiqa": "fed for other document fields than the deployed schema's",
                                "caiq": "not fed, or its last feed did not finish"}
    monkeypatch.setattr(labvespa, "http", lambda *a, **k: pytest.fail("no query on a stale feed"))
    with pytest.raises(labvespa.VespaError, match="^fiqa is fed for other document fields .*run lab vespa up"):
        labvespa.query("fiqa", "select * from fiqa where true", "hybrid")


def test_the_feed_writes_the_declared_fields_and_warns_of_the_rest(monkeypatch):
    ds = labvespa.lab.Dataset("caiq", ["a", "b"], ["T\x10", ""], ["x", "y"], {}, {}, {})
    monkeypatch.setattr(labvespa.lab, "load", lambda name, split="test": ds)
    monkeypatch.setattr(labvespa.lab, "vectors", lambda ds: (np.array([[0.1, -0.2], [0.0, 0.25]], dtype=np.float32), None))
    warned = []
    n, lines = labvespa.documents("caiq", {"id": "string", "title": "string", "dense_int8": "tensor<int8>(x[2])",
                                           "mine": "string"}, warned.append)
    docs = [json.loads(line) for line in lines]
    assert n == 2 and warned == ["caiq.sd field mine: the lab has no values for it, so it stays empty"]
    assert docs[0] == {"put": "id:caiq:caiq::a", "fields": {"id": "a", "title": "T ", "dense_int8": {"values": "339a"}}}
    with pytest.raises(labvespa.VespaError, match="dense_int8 needs int8 cells"):
        labvespa.documents("caiq", {"dense_int8": "tensor<float>(x[2])"}, warned.append)


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
