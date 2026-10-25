"""Your own Vespa for the lab: lab vespa up | deploy | down | status, and query() for vespa/search.py.

Vespa runs in Docker as the container lab-vespa, its ports on 127.0.0.1 only. The documents and every vector come
from the lab's data (lab setup downloads it): the feed computes no vectors, so Vespa needs no GPU.
"""

from __future__ import annotations

import hashlib
import io
import json
import pathlib
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np

import lab

IMAGE = "vespaengine/vespa:8.763.13"
NAME = "lab-vespa"
QUERY_PORT, CONFIG_PORT = 18080, 19071  # on 127.0.0.1: queries and feed, and the config server (deploy)
LIMITS = ("--memory", "16g", "--cpus", "6")  # the container's share of the machine
DATASETS = ("fiqa", "caiq")
SOURCE = "fiqa"  # vespa/app/schemas/fiqa.sd is every dataset's schema; a deploy makes the others from it
APP = lab.HERE / "vespa" / "app"
MARKER = "/opt/vespa/var/lab-fed.json"  # inside the container: what was fed, so a second up skips the feed
FEED_FORMAT = 1  # part of the marker; bump it when what the feed writes changes
INT8_SCALE = 512  # dense_int8 = round(512 · unit vector); the largest value in the data is 0.22, so 111
ILLEGAL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")  # code points a Vespa string field refuses
QUERY_URL, CONFIG_URL = f"http://127.0.0.1:{QUERY_PORT}", f"http://127.0.0.1:{CONFIG_PORT}"
APPLICATION_URL = f"{CONFIG_URL}/application/v2/tenant/default/application/default/environment/prod/region/default/instance/default"


class VespaError(RuntimeError):
    """A short message for the participant."""


# ---- docker and HTTP

def docker(*args: str, timeout: float = 60, stdin: str | None = None) -> str:
    """docker args; its stdout. Fails with docker's own message."""
    try:
        r = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, input=stdin)
    except FileNotFoundError:
        raise VespaError("docker is not installed; install Docker Engine (https://docs.docker.com/engine/install/)") from None
    except subprocess.TimeoutExpired:
        raise VespaError(f"docker {args[0]} did not finish within {timeout:.0f} s") from None
    if r.returncode:
        lines = [line for line in (r.stderr or r.stdout).splitlines() if line.strip() and "--help" not in line]
        detail = lines[-1] if lines else f"exit code {r.returncode}"
        hint = ("; let your user run docker: sudo usermod -aG docker $USER, then log in again"
                if "permission denied" in detail.lower() else
                f"; Vespa needs ports {QUERY_PORT} and {CONFIG_PORT} on 127.0.0.1: ss -ltnp shows what holds them"
                if "already in use" in detail or "already allocated" in detail else
                "; start Docker: sudo systemctl start docker" if "daemon" in detail.lower() else "")
        raise VespaError(f"docker {args[0]} failed: {detail}{hint}")
    return r.stdout


def http(method: str, url: str, body: bytes | None = None, content_type: str = "application/json",
         timeout: float = 30) -> tuple[int, dict]:
    """(status, JSON body) of one request; an HTTP error status is returned, not raised."""
    req = urllib.request.Request(url, body, {"Content-Type": content_type} if body is not None else {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, text = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, text = e.code, e.read()
    try:
        return status, json.loads(text or b"{}")
    except ValueError:
        return status, {"message": text[:500].decode(errors="replace")}


def reachable(url: str, timeout: float = 5) -> dict | None:
    """The JSON at url, or None while nothing answers there or it answers with an error."""
    try:
        status, data = http("GET", url, timeout=timeout)
    except OSError:
        return None
    return data if status == 200 else None


def wait(what: str, ready, timeout: float, progress=None) -> None:
    """Calls ready() every 2 s until it is true; fails after timeout seconds."""
    start = time.time()
    while not ready():
        if time.time() - start > timeout:
            raise VespaError(f"{what} is not ready after {timeout:.0f} s; docker logs {NAME} shows why")
        if progress:
            progress(f"waiting for {what} · {time.time() - start:.0f} s")
        time.sleep(2)


def state() -> str | None:
    """The container's state (running, exited, ...), or None without one."""
    out = docker("ps", "-a", "--filter", f"name=^{NAME}$", "--format", "{{.State}}").strip()
    return out or None


# ---- the application package

def shown(path: pathlib.Path) -> str:
    """path relative to the working directory when it is inside it, so messages stay short."""
    try:
        return str(path.relative_to(pathlib.Path.cwd()))
    except ValueError:
        return str(path)


def app_dir(app: str | pathlib.Path | None) -> pathlib.Path:
    """app, else ./vespa/app, else the repo's vespa/app; it must hold services.xml."""
    path = pathlib.Path(app) if app else next(p for p in (pathlib.Path("vespa/app"), APP) if p.is_dir())
    if not (path / "services.xml").is_file():
        raise VespaError(f"{path} is not a Vespa application: it has no services.xml")
    return path.resolve()


def package(app: pathlib.Path, warn=print) -> bytes:
    """app as a zip for the config server, with every dataset's schema made from fiqa.sd (schema)."""
    out = io.BytesIO()
    made = {f"schemas/{name}.sd" for name in DATASETS if name != SOURCE}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(app.rglob("*")):
            rel = f.relative_to(app).as_posix()
            if rel in made:
                warn(f"{f} is not used: every deploy makes {f.name} from {SOURCE}.sd; edit {SOURCE}.sd only")
            elif f.is_file() and not f.name.startswith("."):
                z.write(f, rel)
        for name in DATASETS:
            if name != SOURCE:
                z.writestr(f"schemas/{name}.sd", schema(app, name))
    return out.getvalue()


def document_block(sd: str, name: str) -> str:
    """The document {...} block of schema text sd, comments removed."""
    text = re.sub(r"#[^\n]*", "", sd)
    m = re.search(rf"\bdocument\s+{name}\b[^{{]*\{{", text)
    if not m:
        raise VespaError(f"{name}.sd has no 'document {name} {{' block")
    depth, i = 1, m.end()
    while depth and i < len(text):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return " ".join(text[m.start():i].split())


def fields(sd: str, name: str) -> dict[str, str]:
    """The document fields of schema text sd: name -> type."""
    return dict(re.findall(r"\bfield\s+(\w+)\s+type\s+([^{]+?)\s*\{", document_block(sd, name)))


def schema(app: pathlib.Path, name: str) -> str:
    """Dataset name's schema: fiqa.sd, for another dataset with only the schema and document renamed."""
    path = app / "schemas" / f"{SOURCE}.sd"
    if not path.is_file():
        raise VespaError(f"{path} is missing: it is the schema of every dataset ({', '.join(DATASETS)})")
    text = path.read_text()
    if name == SOURCE:
        return text
    renamed, n = re.subn(rf"^(\s*(?:schema|document)\s+){SOURCE}\b", rf"\g<1>{name}", text, flags=re.M)
    if n != 2:
        raise VespaError(f"{path} must start one line with 'schema {SOURCE}' and one with 'document {SOURCE}'")
    return renamed


def fingerprint(sd: str, name: str) -> str:
    """What a feed of dataset name depends on: the data, the feed's format and schema sd's document fields."""
    key = f"{lab.DATA_VERSION} {FEED_FORMAT} {document_block(sd, name)}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---- what the feed writes

def cell_type(tensor_type: str) -> str:
    m = re.match(r"tensor<(\w+)>", tensor_type)
    return m.group(1) if m else "float"


def hexed(x: np.ndarray, cells: str) -> str:
    """x as the hex string of Vespa's tensor JSON, in the field's cell type."""
    if cells == "float":
        return x.astype(">f4").tobytes().hex()
    if cells == "double":
        return x.astype(">f8").tobytes().hex()
    if cells == "bfloat16":
        u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(">u2").tobytes().hex()  # rounded to nearest even
    if cells == "int8":
        return np.asarray(x).astype(np.int8).tobytes().hex()
    raise VespaError(f"the feed cannot write {cells} cells")


def late_bits(ds: lab.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The document token vectors of ds as centered sign bits (16 int8 per token), where document i's tokens start,
    and the mean token vector the bits are centered on; kept in data/cache after the first build."""
    tokens = lab.late_tokens(ds)

    def build() -> dict[str, np.ndarray]:
        mean = tokens.mean()
        bits = np.empty((len(tokens.docs), 16), dtype=np.uint8)
        for a in range(0, len(tokens.docs), 1_000_000):
            x = tokens.docs[a:a + 1_000_000].astype(np.float32) * tokens.scale[a:a + 1_000_000, None].astype(np.float32)
            bits[a:a + 1_000_000] = np.packbits(x - mean >= 0, axis=1)
        return dict(bits=bits, mean=mean)

    a = lab._disk_cached(f"late-bits-{lab.CACHE_FORMAT}-{lab.DATA_VERSION}-{ds.name}", build)
    return np.asarray(a["bits"]).view(np.int8), tokens.start, np.asarray(a["mean"])


def documents(name: str, wanted: dict[str, str], warn) -> tuple[int, object]:
    """(count, the feed's JSON lines) for dataset name: every field of wanted (name -> type) the lab has values for.
    warn(message) hears of the fields it leaves empty."""
    ds = lab.load(name)
    floats = lab.vectors(ds)[0] if {"dense_float", "dense_int8"} & set(wanted) else None
    bits = lab.dense_parts(ds)[1].view(np.uint8).view(np.int8) if "dense_bits" in wanted else None
    tokens = lab.late_tokens(ds) if {"late_bits", "late_int8", "late_scale"} & set(wanted) else None
    late = late_bits(ds)[0] if "late_bits" in wanted else None
    start = tokens.start if tokens else None

    def blocks(x: np.ndarray, i: int) -> dict:  # one hex block per token of document i
        return {"blocks": {str(t): x[a].tobytes().hex() for t, a in enumerate(range(start[i], start[i + 1]))}}

    values = {  # field -> (the cell type it needs, or None for any; its value in document i)
        "id": (None, lambda i: ds.ids[i]),
        "title": (None, lambda i: ILLEGAL.sub(" ", ds.titles[i])),
        "body": (None, lambda i: ILLEGAL.sub(" ", ds.bodies[i])),
        "dense_float": (None, lambda i: {"values": hexed(floats[i], cell_type(wanted["dense_float"]))}),
        "dense_int8": ("int8", lambda i: {"values": hexed(np.clip(np.round(floats[i] * INT8_SCALE), -127, 127), "int8")}),
        "dense_bits": ("int8", lambda i: {"values": bits[i].tobytes().hex()}),
        "late_bits": ("int8", lambda i: blocks(late, i)),
        "late_int8": ("int8", lambda i: blocks(tokens.docs, i)),
        "late_scale": (None, lambda i: {"cells": {str(t): float(tokens.scale[a]) for t, a in enumerate(range(start[i], start[i + 1]))}}),
    }
    for f in sorted(set(wanted) - set(values)):
        warn(f"{name}.sd field {f}: the lab has no values for it, so it stays empty")
    for f in sorted(set(wanted) & set(values)):
        if values[f][0] and cell_type(wanted[f]) != values[f][0]:
            raise VespaError(f"{name}.sd field {f} needs {values[f][0]} cells: the lab writes {values[f][0]} values into it")
    feeding = [(f, v) for f, (_, v) in values.items() if f in wanted]

    def lines():
        for i, doc_id in enumerate(ds.ids):
            yield json.dumps({"put": f"id:{name}:{name}::{doc_id}", "fields": {f: v(i) for f, v in feeding}}) + "\n"

    return len(ds.ids), lines()


# ---- up, deploy, feed, down, status

def start(progress=None) -> None:
    """Starts the lab-vespa container (creating it on first use) and waits for its config server."""
    s = state()
    if s == "created":  # docker run made it but could not start it (a port in use); a later start cannot either
        docker("rm", "--force", NAME, timeout=120)
        s = None
    if s is None:
        if progress:
            progress(f"starting {IMAGE} (the first start downloads about 1.5 GB)")
        docker("run", "--detach", "--name", NAME, "--hostname", NAME, *LIMITS,
               "--publish", f"127.0.0.1:{QUERY_PORT}:8080", "--publish", f"127.0.0.1:{CONFIG_PORT}:19071",
               IMAGE, timeout=900)
    elif s != "running":
        docker("start", NAME, timeout=120)
    wait("the config server", lambda: alive() and (reachable(f"{CONFIG_URL}/state/v1/health") or {}).get("status", {})
         .get("code") == "up", 180, progress)


def alive() -> bool:
    """True while the container runs; fails with its last log line once it has stopped."""
    if state() == "running":
        return True
    logs = subprocess.run(["docker", "logs", "--tail", "50", NAME], capture_output=True, text=True, timeout=30)
    lines = [line for line in (logs.stdout + logs.stderr).splitlines() if "panic" in line or "error" in line.lower()]
    raise VespaError(f"{NAME} stopped while starting: {lines[-1].strip()[:300] if lines else 'docker logs ' + NAME}; "
                     f"lab vespa down, then lab vespa up")


def deploy(app: str | pathlib.Path | None = None, progress=None, warn=print) -> dict:
    """Deploys app (default: ./vespa/app, else the repo's) and waits until Vespa serves it. Returns Vespa's answer:
    warnings under "log", and under "configChangeActions" what needs a restart, refeed or reindex."""
    path = app_dir(app)
    if state() != "running":
        raise VespaError("Vespa is not running; lab vespa up starts it")
    for name in DATASETS:
        document_block(schema(path, name), name)
    zipped = package(path, warn)
    if progress:
        progress(f"deploying {path}")
    try:
        status, answer = http("POST", f"{CONFIG_URL}/application/v2/tenant/default/prepareandactivate", zipped,
                              "application/zip", timeout=300)
    except OSError as e:
        raise VespaError(f"cannot reach the config server at {CONFIG_URL}: {e}; lab vespa up starts it") from None
    if status != 200:
        message = str(answer.get("message", answer))
        for name in DATASETS:  # an error in a schema made from fiqa.sd is on the same line of fiqa.sd
            message = message.replace(f"'{name}.sd'", f"'{SOURCE}.sd'")
        raise VespaError(f"Vespa refused {shown(path)}: {brief_deploy_error(message)}")
    wait("the new application", lambda: (reachable(f"{APPLICATION_URL}/serviceconverge") or {}).get("converged") is True,
         300, progress)
    wait("the query endpoint", ready, 300, progress)
    return answer


LONG_VALUE = re.compile(r"'[^']{60,}'|\[[^\]]{60,}\]|\{[^}]{60,}\}")  # a tensor or list Vespa echoes back
MAX_LINE = 300


def cut(text: str) -> str:
    """text with every long quoted value, list or tensor cut to its first 30 characters."""
    return LONG_VALUE.sub(lambda m: f"{m.group(0)[:30]}…{m.group(0)[-1]}", text)


def brief_deploy_error(message: str, lines: int = 8) -> str:
    """Vespa's refusal of an application, its cause first: the first warning (the rank feature or setting that failed),
    then the other lines that are not INFO, each once, at most lines lines."""
    rows = [cut(r.strip())[:MAX_LINE] for r in message.splitlines() if r.strip() and not r.strip().startswith("INFO:")]
    first = next((r for r in rows if r.startswith("WARNING:")), rows[0] if rows else "no reason given")
    out = [first.removeprefix("WARNING: ")]
    for r in rows:
        if r != first and r not in out and len(out) < lines:
            out.append(r)
    return "\n  ".join(out)


def brief_query_error(error: dict) -> str:
    """One of Vespa's query errors, its root cause (the last part of a "a: b: cause" chain) first, values cut short."""
    text = cut(" ".join(str(error.get(k, "")) for k in ("summary", "message")).strip())
    cause = text.rsplit(": ", 1)[-1]
    return f"Vespa: {cause}" + (f"\n  {text[:MAX_LINE]}" if cause != text else "")


def ready() -> bool:
    """Whether the container answers queries with every content node up."""
    try:
        status, data = http("POST", f"{QUERY_URL}/search/", json.dumps(
            {"yql": "select * from sources * where true", "hits": 0, "timeout": "5s"}).encode(), timeout=10)
    except OSError:
        return False
    return status == 200 and not data.get("root", {}).get("errors")


def changes(answer: dict) -> list[str]:
    """Lines on what Vespa said about a deploy: its warnings, and what needs a restart, refeed or reindex."""
    out = [f"{m.get('level', '').lower()}: {m.get('message')}" for m in answer.get("log", [])
           if m.get("level") in ("WARNING", "ERROR")]
    actions = answer.get("configChangeActions", {})
    for kind in ("restart", "refeed", "reindex"):
        for a in actions.get(kind, []):
            what = "; ".join(a.get("messages", [])) or a.get("name", "")
            out.append(f"{kind}: {a.get('documentType', '')} {what}".strip())
    return out


def fed() -> dict[str, str]:
    """dataset -> the fingerprint of its last complete feed in this container."""
    try:
        return json.loads(docker("exec", NAME, "sh", "-c", f"cat {MARKER} 2>/dev/null || echo '{{}}'"))
    except ValueError:
        return {}


def deployed(name: str) -> str | None:
    """The schema of dataset name in the application Vespa runs now, or None without one."""
    try:
        with urllib.request.urlopen(f"{APPLICATION_URL}/content/schemas/{name}.sd", timeout=10) as r:
            return r.read().decode()
    except OSError:  # no application yet, or no config server
        return None


def unfed() -> dict[str, str]:
    """dataset -> why its documents do not hold what the deployed schema declares (not fed yet, or fed for other
    document fields); the datasets that are fine are left out."""
    if state() != "running":
        raise VespaError("Vespa is not running; lab vespa up starts it")
    done, out = fed(), {}
    for name in DATASETS:
        sd = deployed(name)
        if sd is None:
            out[name] = "no application is deployed"
        elif name not in done:
            out[name] = "not fed, or its last feed did not finish"
        elif done[name] != fingerprint(sd, name):
            out[name] = "fed for other document fields than the deployed schema's"
    return out


def feed(name: str, app: pathlib.Path, advance=None, warn=print) -> float:
    """Feeds dataset name with the fields app's schema declares, through the vespa CLI in the container; advance(n)
    hears of the documents sent. Returns the seconds it took."""
    t0 = time.time()
    total, lines = documents(name, fields(schema(app, name), name), warn)
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        proc = subprocess.Popen(["docker", "exec", "-i", NAME, "vespa", "feed", "--target", "local", "-"],
                                stdin=subprocess.PIPE, stdout=out, stderr=err, text=True)
        try:
            for k, line in enumerate(lines, 1):
                proc.stdin.write(line)
                if advance and k % 200 == 0:
                    advance(200)
            proc.stdin.close()
            code = proc.wait(timeout=600)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            code = None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=30)
        out.seek(0)
        err.seek(0)
        stats, errors = out.read(), err.read().strip().splitlines()
    if code != 0 and state() != "running":
        raise VespaError(f"{NAME} stopped while feeding {name}; lab vespa up starts it and feeds again")
    if code != 0:
        raise VespaError(f"feeding {name} failed: " + ("\n".join(errors[-5:]) or f"vespa feed exit code {code}"))
    try:
        ok = json.loads(stats).get("feeder.ok.count")
    except ValueError:
        ok = None
    if ok != total:
        raise VespaError(f"feeding {name}: Vespa took {ok} of {total} documents\n" + "\n".join(errors[-5:]))
    return time.time() - t0


def up(app: str | pathlib.Path | None = None, progress=None, feeding=None, warn=print) -> dict:
    """Starts Vespa, deploys app (default: ./vespa/app, else the repo's) and feeds every dataset whose feed is not in
    the container yet for this data and these document fields. progress(message) hears of each step, and
    feeding(dataset, documents) returns the advance(n) the feed calls. Returns the deploy's answer and the feed times."""
    path = app_dir(app)
    if not (lab.DATA / lab.DATA_VERSION).is_dir():
        raise VespaError("the lab's data is not here yet; run lab setup first")
    start(progress)
    answer = deploy(path, progress, warn)
    done, seconds = fed(), {}
    for name in DATASETS:
        if done.get(name) == fingerprint(schema(path, name), name):
            continue
        done.pop(name, None)  # an interrupted feed leaves the dataset unfed, not fed for its old fields
        docker("exec", "-i", NAME, "sh", "-c", f"cat > {MARKER}", stdin=json.dumps(done))
        advance = feeding(name, len(lab.load(name).ids)) if feeding else None
        seconds[name] = feed(name, path, advance, warn)
        done[name] = fingerprint(schema(path, name), name)
        docker("exec", "-i", NAME, "sh", "-c", f"cat > {MARKER}", stdin=json.dumps(done))
    return dict(deploy=answer, fed=seconds)


def down() -> bool:
    """Removes the lab-vespa container and everything fed into it; False when there was none."""
    if state() is None:
        return False
    docker("rm", "--force", NAME, timeout=120)
    return True


def status() -> dict:
    """The container's state, memory, and per dataset its document count and why its feed is not current (None when
    it is)."""
    s = state()
    out: dict = dict(state=s, image=IMAGE, query=QUERY_URL, config=CONFIG_URL)
    if s != "running":
        return out
    out["memory"] = docker("stats", "--no-stream", "--format", "{{.MemUsage}}", NAME).strip()
    out["serving"] = ready()
    problems, out["datasets"] = unfed(), {}
    for name in DATASETS:
        count = None
        if out["serving"]:
            try:
                _, data = http("POST", f"{QUERY_URL}/search/", json.dumps(
                    {"yql": f"select * from {name} where true", "hits": 0, "timeout": "5s"}).encode(), timeout=10)
                count = data.get("root", {}).get("fields", {}).get("totalCount")
            except OSError:
                pass
        out["datasets"][name] = dict(documents=count, problem=problems.get(name))
    return out


# ---- queries

_LATE: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {}  # dataset -> (question text -> token vectors, mean)
_LATE_ENCODER: list = []  # the late-interaction question encoder, loaded on the first question outside the data


def late_inputs(ds: lab.Dataset, question: str) -> dict:
    """The question's late-interaction tokens as the maxsim profiles' inputs: query(qlate_float), the vectors, and
    query(qlate), their centered sign bits (16 int8 per token). Questions of the data come with their tokens; others
    are encoded with lab.LATE_MODEL (on the GPU if there is one)."""
    if ds.name not in _LATE:
        tokens = lab.late_tokens(ds)
        texts = {json.loads(line)["id"]: json.loads(line)["text"] for line in open(lab.fetch() / ds.name / "queries.jsonl")}
        _LATE[ds.name] = ({texts[q]: v for q, v in tokens.queries.items()}, late_bits(ds)[2])
    known, mean = _LATE[ds.name]
    vectors = known.get(question)
    if vectors is None:
        import torch
        from sentence_transformers import MultiVectorEncoder

        if not _LATE_ENCODER:
            _LATE_ENCODER.append(MultiVectorEncoder(lab.LATE_MODEL, device="cuda" if torch.cuda.is_available() else "cpu"))
        vectors = np.asarray(_LATE_ENCODER[0].encode_query([question])[0].float().cpu(), dtype=np.float32)
    bits = np.packbits(vectors - mean >= 0, axis=1).view(np.int8)
    return {"query(qlate)": {"blocks": {str(t): b.tolist() for t, b in enumerate(bits)}},
            "query(qlate_float)": {"blocks": {str(t): v.tolist() for t, v in enumerate(vectors)}}}


_FED_CHECKED: set[str] = set()  # the datasets this process found fed for the deployed schema


def require_fed(ds_name: str) -> None:
    """Fails unless dataset ds_name's documents hold the deployed schema's fields, so no search scores on a stale or
    half-done feed; checked once per process."""
    if ds_name in _FED_CHECKED:
        return
    problem = unfed().get(ds_name)
    if problem:
        raise VespaError(f"{ds_name} is {problem}: run lab vespa up to feed it (about 1.5 minutes)")
    _FED_CHECKED.add(ds_name)


def query(ds_name: str, yql: str, ranking: str, inputs: dict | None = None, hits: int = 10,
          **params) -> list[tuple[str, float]]:
    """Runs yql on dataset ds_name with rank profile ranking; (doc id, relevance) of the best hits, best first.
    inputs: the rank profile's query(...) inputs, as numbers or lists (numpy arrays are converted); params: other
    request parameters, such as q for userInput(@q). Fails while ds_name's feed does not match the deployed schema."""
    require_fed(ds_name)
    body = {"yql": yql, "ranking": ranking, "hits": hits, "model.restrict": ds_name, "presentation.summary": "ids",
            "timeout": "10s", **params}
    for k, v in (inputs or {}).items():
        body[f"input.{k}"] = v.tolist() if isinstance(v, np.ndarray) else v
    try:
        status, data = http("POST", f"{QUERY_URL}/search/", json.dumps(body).encode(), timeout=30)
    except OSError as e:
        raise VespaError(f"Vespa does not answer at {QUERY_URL} ({getattr(e, 'reason', e)}); lab vespa up starts it") from None
    errors = data.get("root", {}).get("errors") or ([data] if status != 200 else [])
    if errors:
        raise VespaError(brief_query_error(errors[0]))
    return [(h["fields"]["id"], h["relevance"]) for h in data["root"].get("children", [])]
