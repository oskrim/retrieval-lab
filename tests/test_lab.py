"""pytest -p no:xdist tests"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import time
import urllib.error

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


@pytest.mark.parametrize("text, detail", [
    ('{"detail": "at most 10 submissions per hour"}', "at most 10 submissions per hour"),
    ('{"error": {"message": "your model budget of $1.00 is used up", "type": "insufficient_quota"}}',
     "your model budget of $1.00 is used up"),
    ('{"object": "error", "message": "CUDA out of memory", "code": 500}', "CUDA out of memory"),
    ("<html>\n<h1>502 Bad Gateway</h1>\n</html>", "<html> <h1>502 Bad Gateway</h1> </html>"),
    ("", ""),
])
def test_error_detail(text, detail):
    assert lab.error_detail(text) == detail


@pytest.mark.parametrize("failure, message", [
    (lab.LabError(500, "CUDA out of memory"), "answered 500: CUDA out of memory"),
    (urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")), "lab status shows whether it is up"),
    (ValueError("Expecting value"), "Expecting value"),
])
def test_chat_failures_name_the_agent_model(endpoint, failure, message):
    endpoint.setattr(lab, "local_agent_model", lambda: lab.AGENT_MODEL)

    def post(*a, **kw):
        raise failure

    endpoint.setattr(lab, "post", post)
    with pytest.raises(lab.ModelError, match=message):
        lab.chat([dict(role="user", content="hi")])


def test_connect_saves_only_a_working_url_and_token(endpoint):
    saved = json.dumps(dict(url="https://lab.example", token="good"))
    lab.LAB.write_text(saved)

    def post(url, body, headers, timeout=120):
        if headers["Authorization"] != "Bearer good":
            raise lab.LabError(401, "send Authorization: Bearer <the token from your lab page>")
        return dict(nick="ada")

    endpoint.setattr(lab, "post", post)
    with pytest.raises(RuntimeError, match="refused the token"):
        lab.connect("https://lab.example", "typo")
    with pytest.raises(RuntimeError, match="not a lab URL"):
        lab.connect("lab.example", "good")
    assert lab.LAB.read_text() == saved
    assert lab.connect(" https://other.example/ ", " good\n") == "ada"
    assert json.loads(lab.LAB.read_text()) == dict(url="https://other.example", token="good")


def test_parallel_map_does_not_wait_for_running_items_after_an_error():
    release = threading.Event()

    def fn(x):
        if x == 0:
            raise ValueError("boom")
        release.wait(30)  # an agent waiting on the model
        return x

    start = time.time()
    try:
        with pytest.raises(ValueError, match="boom"):
            lab.parallel_map(fn, [0, 1], threads=2)
        assert time.time() - start < 5
    finally:
        release.set()
    assert lab.parallel_map(lambda x: 2 * x, [1, 2, 3]) == [2, 4, 6]


@pytest.fixture
def fake_model(monkeypatch):
    """An OpenAI-compatible server that answers with the scripted messages in turn and keeps the requests."""
    script, requests = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(dict(choices=[dict(index=0, message=script[len(requests) - 1])])).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(lab, "_AGENT", [(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", {}, "fake")])
    yield script, requests
    server.shutdown()


def test_crm_agent_takes_null_content_and_null_arguments(fake_model):
    script, requests = fake_model
    args = dict(query=None, filters=dict(country="Germany", stage="Closed lost"), limit=None, descending=None)
    script += [dict(role="assistant", content=None, tool_calls=[dict(id="c1", type="function", function=dict(
                   name="search", arguments=json.dumps(args)))]),
               dict(role="assistant", content=None, reasoning_content="nothing to add")]
    out = lab.crm_agent("Which deals did we lose in Germany?", [lab.ATTR_SEARCH], lab.ATTR)
    assert out == dict(answer="", deals=set(), calls=["search"])
    result = json.loads(requests[1]["messages"][-1]["content"])
    assert result["total"] == sum(d["country"] == "Germany" and d["stage"] == "Closed lost" for d in lab.DEALS)
    assert len(result["deals"]) == min(10, result["total"])


def test_crm_agent_shows_the_model_a_wrong_argument_type(fake_model):
    script, requests = fake_model
    script += [dict(role="assistant", content=None, tool_calls=[dict(id="c1", type="function", function=dict(
                   name="search", arguments=json.dumps(dict(filters=["country"]))))]),
               dict(role="assistant", content="DEALS: none")]
    assert lab.crm_agent("Deals in Germany?", [lab.ATTR_SEARCH], lab.ATTR)["deals"] == set()
    assert "AttributeError" in requests[1]["messages"][-1]["content"]


def test_an_index_read_back_from_the_disk_cache_scores_as_a_fresh_build(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "DATA", tmp_path)
    monkeypatch.setattr(lab, "PERSIST_CHARS", 0)
    texts = ["Encryption of data at rest", "Data is encrypted in transit and at rest", "Backups run nightly"]
    analyzer = lab.Analyzer(stopwords=True, stemming=True)

    def fresh(idf_texts):
        for memo in ("_POSTINGS", "_DF", "_indexes"):  # a new process: nothing in memory
            monkeypatch.setattr(lab, memo, {})
        return lab.Index(texts, analyzer, idf_texts)

    idf_texts = texts + ["encryption keys"]
    built = fresh(idf_texts)
    assert len(list((tmp_path / "cache").iterdir())) == 2  # the postings and the idf document frequencies
    read = fresh(idf_texts)
    assert isinstance(read.keys, np.memmap)
    assert read.vocab == built.vocab
    for q in ("encrypted data at rest", "nightly backups", "unknown words"):
        assert read.bm25(q).tolist() == built.bm25(q).tolist()
        assert read.phrase(q).tolist() == built.phrase(q).tolist()


def test_an_analyzer_subclass_keeps_its_index_in_memory_only(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "DATA", tmp_path)
    monkeypatch.setattr(lab, "PERSIST_CHARS", 0)

    class Upper(lab.Analyzer):
        def word(self, token):
            return token.upper()

    assert lab.Index(["a b", "b c"], Upper()).vocab == {"A": 0, "B": 1, "C": 2}
    assert not (tmp_path / "cache").exists()


def test_packed_sign_bits():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 100)).astype(np.float32)
    bits = lab.pack_signs(x >= 0)
    assert bits.dtype == np.uint64 and bits.shape == (5, 2)
    signs = np.where(x >= 0, 1, -1).astype(np.float32)
    assert np.array_equal(lab.unpack_signs(bits, 100), signs)
    assert np.array_equal(lab.unpack_signs(bits[[3, 1]], 100), signs[[3, 1]])
    q = rng.normal(size=100).astype(np.float32)
    agree = 100 - 2 * np.bitwise_count(bits ^ lab.pack_signs(q >= 0)).sum(1, dtype=np.int64)
    assert agree.tolist() == (signs @ np.where(q >= 0, 1, -1)).tolist()


def test_no_cuda_devices_means_no_torch(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setitem(__import__("sys").modules, "torch", None)  # importing torch would raise
    assert lab._cuda() is False
