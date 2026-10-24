"""pytest -p no:xdist tests"""

import json

import numpy as np
import pytest

import lab


@pytest.fixture
def no_gpu(tmp_path, monkeypatch):
    """A three-document dataset, a shipped cache that scores two of its documents for one question, and no GPU."""
    monkeypatch.setattr(lab, "DATA", tmp_path)
    monkeypatch.setattr(lab, "_RERANK_SHIPPED", {})
    monkeypatch.setattr(lab, "_RERANK_DB", [])
    monkeypatch.setattr(lab, "rerank_misses", [0])
    monkeypatch.setattr(lab, "_cuda", lambda: False)
    monkeypatch.setattr(lab, "_reranker", lambda *a: pytest.fail("the model must not load without a GPU"))
    (tmp_path / lab.DATA_VERSION).mkdir()
    np.savez(tmp_path / lab.DATA_VERSION / lab.rerank_cache_name(), questions=np.array([lab.question_key("q")]),
             offsets=np.array([0, 2]), docs=np.array(["a", "b"]), scores=np.array([2.5, -1.0], dtype=np.float32))
    return lab.Dataset("tiny", ["a", "b", "c"], ["", "", ""], ["x", "y", "z"], {}, {}, {})


def test_rerank_local_reads_the_shipped_cache(no_gpu):
    assert lab.rerank_local(no_gpu, "q", [1, 0]).tolist() == [-1.0, 2.5]
    assert lab.rerank_misses[0] == 0


def test_without_a_gpu_a_miss_scores_lowest_of_the_call_and_warns_once(no_gpu, capsys):
    assert lab.rerank_local(no_gpu, "q", [2, 0]).tolist() == [2.5, 2.5]
    assert lab.rerank_local(no_gpu, "q", [0, 1, 2]).tolist() == [2.5, -1.0, -1.0]
    assert lab.rerank_local(no_gpu, "unknown", [0, 1]).tolist() == [0.0, 0.0]
    assert lab.rerank_misses[0] == 4
    assert capsys.readouterr().err.count("rerank_local:") == 1


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "_AGENT", [])
    monkeypatch.setattr(lab, "LAB", tmp_path / ".lab.json")
    return monkeypatch


def test_chat_uses_the_local_model_when_it_serves_agent_model(endpoint, capsys):
    endpoint.setattr(lab, "local_agent_model", lambda: lab.AGENT_MODEL)
    assert lab.agent_endpoint() == (f"{lab.VLLM_URL}/chat/completions", {}, lab.AGENT_MODEL)
    assert "on this machine" in capsys.readouterr().err


@pytest.mark.parametrize("local", [None, "some/other-model"])
def test_chat_falls_back_to_the_lab_and_says_so_once(endpoint, capsys, local):
    endpoint.setattr(lab, "local_agent_model", lambda: local)
    lab.LAB.write_text(json.dumps(dict(url="https://lab.example", token="t0k")))
    sent = []
    endpoint.setattr(lab, "post", lambda url, body, headers, timeout: sent.append((url, body, headers)) or {})
    lab.chat([dict(role="user", content="hi")])
    lab.chat([dict(role="user", content="again")])
    assert [(u, b["model"], h) for u, b, h in sent] == 2 * [("https://lab.example/v1/chat/completions", lab.LAB_AGENT_MODEL,
                                                              {"Authorization": "Bearer t0k"})]
    assert capsys.readouterr().err.count("through the lab") == 1


def test_chat_without_a_model_or_the_lab_says_what_to_do(endpoint):
    endpoint.setattr(lab, "local_agent_model", lambda: None)
    with pytest.raises(RuntimeError, match="lab serve.*lab connect"):
        lab.agent_endpoint()
