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


def test_without_a_gpu_a_miss_scores_lowest_of_the_call_and_warns_once(no_gpu):
    with pytest.warns(lab.RerankCacheMiss) as caught:
        assert lab.rerank_local(no_gpu, "q", [2, 0]).tolist() == [2.5, 2.5]
        assert lab.rerank_local(no_gpu, "q", [0, 1, 2]).tolist() == [2.5, -1.0, -1.0]
        assert lab.rerank_local(no_gpu, "unknown", [0, 1]).tolist() == [0.0, 0.0]
    assert lab.rerank_misses[0] == 4
    assert len(caught) == 1


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
    assert capsys.readouterr().err.count("through the API") == 1


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


def test_chat_waits_while_the_lab_server_does_not_answer(endpoint, capsys):
    endpoint.setattr(lab, "local_agent_model", lambda: None)
    lab.LAB.write_text(json.dumps(dict(url="https://lab.example", token="t")))
    endpoint.setattr(lab, "LAB_WAITS", (0, 0, 0))
    failures = [lab.LabError(502, "Bad Gateway"), urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))]
    calls = []

    def post(*a, **kw):
        calls.append(a)
        if failures:
            raise failures.pop(0)
        return dict(choices=[dict(message=dict(role="assistant", content="hi"))])

    endpoint.setattr(lab, "post", post)
    assert lab.chat([dict(role="user", content="hi")])["choices"] and len(calls) == 3
    err = capsys.readouterr().err
    assert "lab server unreachable (502 Bad Gateway), retrying in 0 s" in err and "Connection refused), retrying" in err
    failures.extend([lab.LabError(503, "Service Unavailable")] * 4)
    with pytest.raises(lab.ModelError, match=r"lab server unreachable \(503 Service Unavailable\) after 4 tries"):
        lab.chat([dict(role="user", content="hi")])
    calls.clear()
    failures.append(lab.LabError(429, "your model budget of $1.00 is used up"))  # an answer: not tried again
    with pytest.raises(lab.ModelError, match="answered 429"):
        lab.chat([dict(role="user", content="hi")])
    assert len(calls) == 1


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
    assert json.loads(lab.LAB.read_text()) == dict(url="https://other.example", token="good", tz=None)  # an older lab


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
            message = dict(script[len(requests) - 1])
            usage = {"usage": message.pop("usage")} if "usage" in message else {}
            body = json.dumps(dict(choices=[dict(index=0, message=message)], **usage)).encode()
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


def test_an_agent_of_python_functions_describes_them_to_the_model_and_runs_them(fake_model):
    script, requests = fake_model

    def deals_in(country: str, stages: list[str] | None = None, filters: dict | None = None) -> list[str]:
        """The deal ids in country."""
        return [d["deal_id"] for d in lab.DEALS if lab.matches(d, "country", country)][:2]

    script += [dict(role="assistant", content=None, tool_calls=[dict(id="c1", type="function", function=dict(
                   name="deals_in", arguments=json.dumps(dict(country="germany"))))]),
               dict(role="assistant", content="DEALS: D0007")]
    assert lab.agent("Deals in Germany?", [deals_in]) == {"D0007"}
    schema = requests[0]["tools"][0]["function"]
    assert schema["name"] == "deals_in" and schema["description"] == "The deal ids in country."
    assert schema["parameters"] == dict(type="object", required=["country"], properties=dict(
        country=dict(type="string"), stages=dict(type="array", items=dict(type="string")),
        filters=dict(type="object", additionalProperties=True)))
    germany = [d["deal_id"] for d in lab.DEALS if d["country"] == "Germany"][:2]
    assert json.loads(requests[1]["messages"][-1]["content"]) == germany


def test_crm_agent_shows_the_model_a_wrong_argument_type(fake_model):
    script, requests = fake_model
    script += [dict(role="assistant", content=None, tool_calls=[dict(id="c1", type="function", function=dict(
                   name="search", arguments=json.dumps(dict(filters=["country"]))))]),
               dict(role="assistant", content="DEALS: none")]
    assert lab.crm_agent("Deals in Germany?", [lab.ATTR_SEARCH], lab.ATTR)["deals"] == set()
    assert "AttributeError" in requests[1]["messages"][-1]["content"]


def tool_call(id_: str, name: str, **args) -> dict:
    return dict(id=id_, type="function", function=dict(name=name, arguments=json.dumps(args)))


def test_tracing_records_each_turn_and_a_summary_of_what_each_tool_returned(fake_model, monkeypatch):
    script, _ = fake_model
    monkeypatch.setattr(lab, "CRM_TURNS", 2)  # the third call turns tools off: a user message follows the tool results
    functions = dict(find_values=lambda text, field=None: [dict(field="industry", value="Pharmaceuticals")],
                     search=lambda filters, limit=None: dict(total=2, deals=[dict(deal_id="D0001", company="Novex Pharma"),
                                                                             dict(deal_id="D0002", company="Zenix Labs")]))
    script += [dict(role="assistant", content="Looking up the industry.", usage=dict(total_tokens=100, usd=0.001),
                    tool_calls=[tool_call("c1", "find_values", text="pharma", field="industry")]),
               dict(role="assistant", content=None, usage=dict(total_tokens=300, usd=0.002), tool_calls=[
                   tool_call("c2", "search", filters=dict(industry="Pharmaceuticals", employees=dict(min=2001)), limit=None)]),
               dict(role="assistant", content="Two deals.\nDEALS: D0001, D0002", usage=dict(total_tokens=400, usd=0.003))]
    with lab.tracing() as turns:
        assert lab.crm_agent("Pharma with 2,000+ staff?", [], functions)["deals"] == {"D0001", "D0002"}
    assert turns == [
        dict(calls=[dict(id="c1", name="find_values", args=dict(text="pharma", field="industry"),
                         result="industry: Pharmaceuticals")], text="Looking up the industry.", tokens=100, usd=0.001),
        dict(calls=[dict(id="c2", name="search", args=dict(filters=dict(industry="Pharmaceuticals", employees=dict(min=2001)),
                                                           limit=None),
                         result="2 deals: D0001 Novex Pharma, D0002 Zenix Labs")], text="", tokens=300, usd=0.002),
        dict(calls=[], text="Two deals. DEALS: D0001, D0002", tokens=400, usd=0.003)]
    script.append(dict(role="assistant", content="DEALS: none"))
    lab.crm_agent("Untraced?", [], functions)  # outside tracing nothing is kept
    assert len(turns) == 3


@pytest.mark.parametrize("content, summary", [
    (json.dumps(dict(total=48, deals=[dict(deal_id="D0387", company="Zenith Group", stage="Proposal")],
                     counts={"Germany": 30, "France": 18})), "48 deals (1 shown): D0387 Zenith Group · counts Germany 30, France 18"),
    (json.dumps(dict(total=0, deals=[])), "0 deals: none"),
    (json.dumps([dict(field="industry", value="Pharmaceuticals"), dict(field="industry", value="Healthcare"),
                 dict(field="stage", value="Proposal")]), "industry: Pharmaceuticals, Healthcare · stage: Proposal"),
    (json.dumps([dict(field="amount_eur", about="the size of the deal"), dict(field="close_date", about="when")]),
     "amount_eur · close_date"),
    ("1. [doc:D0001] Acme Oy\n   wants a data platform\n2. [doc:D0002] Beta AB\n   needs SSO", "2 deals: D0001 Acme Oy, D0002 Beta AB"),
    ("[doc:D0001] Acme Oy\nindustry: Retail", "1 deal: D0001 Acme Oy"),
    (json.dumps(dict(error="TypeError: search() got an unexpected keyword argument 'x'")),
     "error: TypeError: search() got an unexpected keyword argument 'x'"),
    ("no document D9999", "no document D9999"),
    ("word " * 100, ("word " * 60)[:lab.TRACE_CHARS - 1].rstrip() + "…"),
])
def test_summarize_puts_a_tool_result_on_one_short_line(content, summary):
    assert lab.summarize(content) == summary


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


@pytest.fixture
def graded():
    """Four CAIQ-like documents (one placeholder title, one empty) and a question whose relevant documents have grades
    2 and 1."""
    return lab.Dataset("caiq", ["a", "b", "c", "d"], ["Acme — X-1: Is data encrypted?", "Untitled", "", "Acme — X-2: Keys?"],
                       ["Yes. AES-256 at rest.", "Partial. " + "word " * 30, "&gt; Not  applicable", "Yes."],
                       {"q": "Is data encrypted at rest?"}, {"q": {"a": 2, "d": 1}}, {"q": set()})


def test_cut_makes_one_line_and_cuts_with_an_ellipsis():
    assert lab.cut(" a\n b &amp;  c ", 10) == "a b & c"
    assert lab.cut("abcdefghij", 6) == "abcde…"


def test_snippet_leaves_out_placeholder_titles(graded):
    assert lab.snippet(graded, 0) == "Acme — X-1: Is data encrypted? Yes. AES-256 at rest."
    assert lab.snippet(graded, 1, 20) == "Partial. word word…"
    assert lab.snippet(graded, 2) == "> Not applicable"


def test_show_marks_relevance_and_the_grade_when_grades_differ(graded, capsys):
    lab.show(graded, "q", {"bm25": np.array([[0.1, 0.9, 0.5, 0.3]], dtype=np.float32)}, k=3)
    assert capsys.readouterr().out == (
        "Question  Is data encrypted at rest?\n"
        "          ✓✓ most relevant (grade 2), ✓ relevant (grade 1)\n\n"
        "bm25 · nDCG@10 0.494 · 1 of the top 3 relevant\n"
        "  1  ·   Partial. word word word word word word word word word word word…\n"
        "  2  ·   > Not applicable\n"
        "  3  ✓   Acme — X-2: Keys? Yes.\n\n")


def test_ranked_lists_mark_relevant_documents_with_one_check_when_grades_are_equal(graded, capsys):
    graded.qrels["q"] = {"a": 1, "d": 1}
    lab.print_ranked(graded, "q", [0, 3], "dense")
    assert capsys.readouterr().out == "dense · all of the top 2 relevant\n  1  ✓  Acme — X-1: Is data encrypted? Yes. AES-256 at rest.\n  2  ✓  Acme — X-2: Keys? Yes.\n"
    lab.print_ranked(graded, "q", [1], "dense")
    assert capsys.readouterr().out.startswith("dense · none of the top 1 relevant\n  1  ·  Partial.")


def test_example_leads_with_the_scores_and_says_jumps_in_words(capsys):
    ds = lab.Dataset("fiqa", ["d"], [""], ["x"], {"q1": "a", "q2": "b", "q3": "c"}, {}, {})
    before, after = {"q1": 0.1, "q2": 0.5, "q3": 0.5}, {"q1": 0.9, "q2": 0.4, "q3": 0.5}
    assert lab.example(ds, before, after) == "q1"
    assert capsys.readouterr().out == (
        "FiQA  nDCG@10 0.367 → 0.600  ·  1 question better, 1 worse (of 3)  ·  1 jump from below 0.3 to above 0.7\n\n")


def test_progress_names_the_datasets(capsys):
    lab.print_progress({"caiq": {"bm25": {"q": 0.5}}, "fiqa": {"bm25": {"q": 0.25}}})
    assert capsys.readouterr().out.split("\n")[:2] == [
        "            CAIQ               FiQA         ", "bm25       0.500              0.250         "]


def test_submit_sends_the_dev_score_to_the_live_board(monkeypatch):
    dev = lab.Dataset("fiqa", ["d1", "d2", "d3"], ["", "", ""], ["", "", ""], {"q1": "a?", "q2": "b?"},
                      {"q1": {"d1": 1}, "q2": {"d3": 1}}, {"q1": set(), "q2": {"d1"}})
    monkeypatch.setattr(lab, "load", lambda name, split: dev if (name, split) == ("fiqa", "dev") else pytest.fail(split))
    sent = []
    monkeypatch.setattr(lab, "call", lambda path, body: sent.append((path, body)) or dict(score=next(iter(body["scores"].values())), rank=1, of=1))
    assert lab.submit(dev, lambda ds, question, k: [0, 2], "mine")["score"] == 1.0  # q2's hidden d1 is skipped
    assert sent == [("/api/board", dict(label="mine", scores={"fiqa dev": 1.0}))]
    with pytest.raises(ValueError, match="FiQA"):
        lab.submit(lab.Dataset("caiq", [], [], [], {}, {}, {}), lambda ds, question, k: [], "x")
    monkeypatch.setattr(lab, "crm_dev", lambda: [("x?", {"D1"}), ("y?", set())])
    assert lab.submit_agent(lambda q: {"D1"}, "agent") == dict(score=0.5, rank=1, of=1)
    assert sent[-1] == ("/api/agent-board", dict(label="agent", scores={"crm dev": 0.5}))
