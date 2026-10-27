"""Your own Vespa for the lab: lab vespa up | deploy | down | status, and query() for search/search.py.

Vespa runs in Docker as the container lab-vespa, its ports on 127.0.0.1 only. A search folder (the lab's search/)
holds the application (app/) and feed.py, whose documents(ds, fields) says what each dataset's documents hold; this
module deploys the one and feeds what the other yields.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
import zipfile

import numpy as np

import lab

IMAGE = "vespaengine/vespa:8.763.13"
NAME = "lab-vespa"
QUERY_PORT, CONFIG_PORT = 18080, 19071  # on 127.0.0.1: queries and feed, and the config server (deploy)
def limits() -> tuple[str, ...]:
    """The container's share of the machine: up to 6 CPUs and 16 GB, and never more than a small box has."""
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return "--memory", f"{min(16, max(2, int(ram_gb * 0.75)))}g", "--cpus", str(min(6, os.cpu_count() or 1))
DATASETS = ("fiqa", "caiq")
TEMPLATE, PLACEHOLDER = "schema.sd", "dataset"  # app/schema.sd is every dataset's schema, named "dataset"
FOLDER = pathlib.Path("search")  # the participant's search folder, relative to the working directory
LAB_FOLDER = lab.HERE / "search"  # the lab's search folder, used when the working directory has none
MARKER = "/opt/vespa/var/lab-fed.json"  # inside the container: what was fed, so a second up skips the feed
APP_MARKER = "/opt/vespa/var/lab-app.txt"  # inside the container: the digest of the app/ deployed last
FEED_FORMAT = 3  # part of the marker; bump it when how the feed writes or how Vespa indexes it changes
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
                "; start Docker: sudo systemctl start docker" if "cannot connect to the docker daemon" in detail.lower() else "")
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


def search_folder(folder: str | pathlib.Path | None = None) -> pathlib.Path:
    """folder, else ./search, else the lab's template; it must hold app/services.xml and feed.py."""
    path = pathlib.Path(folder) if folder else FOLDER if FOLDER.is_dir() else LAB_FOLDER
    if (path / "services.xml").is_file():
        raise VespaError(f"{shown(path)} is an application; give the search folder that holds it as app/ and feed.py")
    if not (path / "app" / "services.xml").is_file():
        raise VespaError(f"{shown(path)} is not a search folder: it has no app/services.xml")
    if not (path / "feed.py").is_file():
        raise VespaError(f"{shown(path)} has no feed.py, which says what to feed")
    return path.resolve()


def contents(app: pathlib.Path, warn=print) -> dict[str, bytes]:
    """The files of app as deployed (path -> bytes), with every dataset's schema made from schema.sd (schema)."""
    out: dict[str, bytes] = {}
    made = {f"schemas/{name}.sd" for name in DATASETS}
    for f in sorted(app.rglob("*")):
        rel = f.relative_to(app).as_posix()
        if rel in made:
            warn(f"{shown(f)} is not used: every deploy makes {f.name} from {TEMPLATE}; edit {TEMPLATE} only")
        elif f.is_file() and not f.name.startswith(".") and rel != TEMPLATE:
            out[rel] = f.read_bytes()
    for name in DATASETS:
        out[f"schemas/{name}.sd"] = schema(app, name).encode()
    return out


def package(app: pathlib.Path, warn=print) -> bytes:
    """app as a zip for the config server (contents)."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, data in contents(app, warn).items():
            z.writestr(rel, data)
    return out.getvalue()


def app_digest(app: pathlib.Path) -> str:
    """A digest of what a deploy of app sends."""
    h = hashlib.sha256()
    for rel, data in sorted(contents(app, lambda message: None).items()):
        h.update(f"\0{rel}\0{len(data)}\0".encode() + data)
    return h.hexdigest()[:16]


def document_block(sd: str, name: str) -> str:
    """The document {...} block of schema text sd, comments removed."""
    text = re.sub(r"#[^\n]*", "", sd)
    m = re.search(rf"\bdocument\s+{name}\b[^{{]*\{{", text)
    if not m:
        raise VespaError(f"the schema of {name} has no 'document {name} {{' block")
    depth, i = 1, m.end()
    while depth and i < len(text):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return " ".join(text[m.start():i].split())


def fields(sd: str, name: str) -> dict[str, str]:
    """The document fields of schema text sd: name -> type."""
    return dict(re.findall(r"\bfield\s+(\w+)\s+type\s+([^{]+?)\s*\{", document_block(sd, name)))


def schema(app: pathlib.Path, name: str) -> str:
    """Dataset name's schema: schema.sd with the schema and the document named name."""
    path = app / TEMPLATE
    if not path.is_file():
        raise VespaError(f"{path} is missing: it is the schema of every dataset ({', '.join(DATASETS)})")
    renamed, n = re.subn(rf"^(\s*(?:schema|document)\s+){PLACEHOLDER}\b", rf"\g<1>{name}", path.read_text(), flags=re.M)
    if n != 2:
        raise VespaError(f"{path} must start one line with 'schema {PLACEHOLDER}' and one with 'document {PLACEHOLDER}'")
    return renamed


def fingerprint(sd: str, name: str, feed_code: str) -> str:
    """What a feed of dataset name depends on: the data, the feed's format, schema sd's document fields and the feed's
    code (feed_digest)."""
    key = f"{lab.DATA_VERSION} {FEED_FORMAT} {feed_code} {document_block(sd, name)}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def feed_digest(folder: pathlib.Path) -> str:
    """A digest of folder's feed.py and of the modules of folder it imports, directly or through each other."""
    h, todo, seen = hashlib.sha256(), ["feed"], set()
    while todo:
        module = todo.pop()
        path = folder / f"{module}.py"
        if module in seen or not path.is_file():
            continue
        seen.add(module)
        code = path.read_bytes()
        h.update(f"\0{module}\0{len(code)}\0".encode() + code)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue  # importing it fails, with the line, when the feed runs
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                todo += [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                todo.append(node.module.split(".")[0])
    return h.hexdigest()[:16]


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

    a = lab.disk_cached(f"late-bits-{lab.CACHE_FORMAT}-{lab.DATA_VERSION}-{ds.name}", build)
    return np.asarray(a["bits"]).view(np.int8), tokens.start, np.asarray(a["mean"])


def load_feed(folder: pathlib.Path):
    """folder's feed.py as a module; folder goes on sys.path, so the feed imports its helpers there (analyze.py)."""
    path = folder / "feed.py"
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("lab_feed", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise VespaError(f"{shown(path)} raised {raised(e, folder)}") from None
    if not callable(getattr(module, "documents", None)):
        raise VespaError(f"{shown(path)} defines no documents(ds, fields)")
    return module


def raised(e: BaseException, folder: pathlib.Path) -> str:
    """The exception in one line, with the innermost line of folder's code it came from."""
    frames = [f for f in traceback.extract_tb(e.__traceback__) if pathlib.Path(f.filename).resolve().is_relative_to(folder)]
    if isinstance(e, SyntaxError) and e.filename and pathlib.Path(e.filename).resolve().is_relative_to(folder):
        frames.append(traceback.FrameSummary(e.filename, e.lineno or 0, "<module>"))
    where = f" ({pathlib.Path(frames[-1].filename).name}:{frames[-1].lineno})" if frames else ""
    return f"{type(e).__name__}: {e}{where}"


def lines(name: str, folder: pathlib.Path, ds: lab.Dataset, wanted: dict[str, str], warn):
    """The feed's JSON lines for dataset name: feed.py's documents(ds, wanted), checked. warn(message) hears of the
    schema's fields the first document leaves empty."""
    path, first = folder / "feed.py", True
    try:
        for item in load_feed(folder).documents(ds, wanted):
            if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], dict)):
                raise VespaError(f"{shown(path)}: documents() yields (document id, fields) pairs; it yielded "
                                 f"{cut(repr(item))[:200]}")
            doc_id, values = item
            extra = set(values) - set(wanted)
            if extra:
                raise VespaError(f"{shown(path)} writes {', '.join(sorted(extra))} into document {doc_id}, which "
                                 f"{TEMPLATE}'s document block does not declare")
            if first:
                for f in sorted(set(wanted) - set(values)):
                    warn(f"{TEMPLATE} field {f}: feed.py writes no value into it, so it stays empty ({name})")
                first = False
            clean = {f: ILLEGAL.sub(" ", v) if isinstance(v, str) else v for f, v in values.items()}
            try:
                yield json.dumps({"put": f"id:{name}:{name}::{doc_id}", "fields": clean}) + "\n"
            except (TypeError, ValueError) as e:
                raise VespaError(f"{shown(path)}: document {doc_id} is not JSON: {e}") from None
    except VespaError:
        raise
    except Exception as e:
        raise VespaError(f"{shown(path)} raised {raised(e, folder)}") from None


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
        docker("run", "--detach", "--name", NAME, "--hostname", NAME, *limits(),
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


def deploy(folder: str | pathlib.Path | None = None, progress=None, warn=print) -> dict:
    """Deploys the app/ of a search folder (default: ./search, else the lab's) and waits until Vespa serves it. Returns
    Vespa's answer: warnings under "log", and under "configChangeActions" what needs a restart, refeed or reindex."""
    path = search_folder(folder) / "app"
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
        for name in DATASETS:  # an error in a schema made from schema.sd is on the same line of schema.sd
            message = message.replace(f"'{name}.sd'", f"'{TEMPLATE}'")
        raise VespaError(f"Vespa refused {shown(path)}: {brief_deploy_error(message)}")
    emptied = {a.get("documentType") for a in answer.get("configChangeActions", {}).get("refeed", [])}
    if emptied & set(fed()):  # a changed field type empties the field, even when a later deploy changes it back
        mark({k: v for k, v in fed().items() if k not in emptied})
    wait("the new application", lambda: (reachable(f"{APPLICATION_URL}/serviceconverge") or {}).get("converged") is True,
         300, progress)
    wait("the query endpoint", ready, 300, progress)
    docker("exec", "-i", NAME, "sh", "-c", f"cat > {APP_MARKER}", stdin=app_digest(path))
    return answer


def sync(folder: str | pathlib.Path | None = None, progress=None, warn=print) -> dict | None:
    """Makes Vespa serve the search folder's app/ (default: ./search, else the lab's) before its search runs: deploys
    app/ when it changed since the last deploy and returns Vespa's answer, or None when Vespa serves it already. Fails,
    naming lab vespa up, when a dataset's documents need feeding again: not fed, or fed for other document fields or
    another feed.py."""
    path = search_folder(folder)
    if state() != "running":
        raise VespaError("Vespa is not running; lab vespa up starts it and feeds the data (about 2 minutes)")
    done, code = fed(), feed_digest(path)
    stale = [name for name in DATASETS if done.get(name) != fingerprint(schema(path / "app", name), name, code)]
    if stale:
        one = len(stale) == 1
        why = (f"{'is' if one else 'are'} not fed" if all(name not in done for name in stale) else
               f"{'was' if one else 'were'} fed for other document fields in {shown(path / 'app' / TEMPLATE)} or "
               f"another {shown(path / 'feed.py')}")
        raise VespaError(f"{' and '.join(stale)} {why}: run lab vespa up to feed again (about 2 minutes)")
    deployed_app = docker("exec", NAME, "sh", "-c", f"cat {APP_MARKER} 2>/dev/null || true").strip()
    if deployed_app == app_digest(path / "app"):
        return None
    return deploy(path, progress, warn)


LONG_VALUE = re.compile(r"'[^']{60,}'|\[[^\]]{60,}\]|\{[^}]{60,}\}")  # a tensor or list Vespa echoes back
MAX_LINE = 300


def cut(text: str) -> str:
    """text with every long quoted value, list or tensor cut to its first 30 characters."""
    return LONG_VALUE.sub(lambda m: f"{m.group(0)[:30]}…{m.group(0)[-1]}", text)


def brief_deploy_error(message: str) -> str:
    """Vespa's refusal of an application, its cause first: the first warning (the rank feature or setting that failed),
    then the other lines that are not INFO, each once, at most 8 lines."""
    rows = [cut(r.strip())[:MAX_LINE] for r in message.splitlines() if r.strip() and not r.strip().startswith("INFO:")]
    first = next((r for r in rows if r.startswith("WARNING:")), rows[0] if rows else "no reason given")
    out = [first.removeprefix("WARNING: ")]
    for r in rows:
        if r != first and r not in out and len(out) < 8:
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


def mark(done: dict[str, str]) -> None:
    """Records done (dataset -> the fingerprint of its complete feed) in the container."""
    docker("exec", "-i", NAME, "sh", "-c", f"cat > {MARKER}", stdin=json.dumps(done))


def deployed(name: str) -> str | None:
    """The schema of dataset name in the application Vespa runs now, or None without one."""
    try:
        with urllib.request.urlopen(f"{APPLICATION_URL}/content/schemas/{name}.sd", timeout=10) as r:
            return r.read().decode()
    except OSError:  # no application yet, or no config server
        return None


def unfed(folder: str | pathlib.Path | None = None) -> dict[str, str]:
    """dataset -> why its documents do not hold what the deployed schema and the search folder's feed.py (default:
    ./search, else the lab's) say (not fed yet, or fed for other document fields or another feed.py); the datasets that
    are fine are left out."""
    if state() != "running":
        raise VespaError("Vespa is not running; lab vespa up starts it")
    path = search_folder(folder)
    done, code, out = fed(), feed_digest(path), {}
    for name in DATASETS:
        sd = deployed(name)
        if sd is None:
            out[name] = "no application is deployed"
        elif name not in done:
            out[name] = "not fed, or its last feed did not finish"
        elif done[name] != fingerprint(sd, name, code):
            out[name] = f"fed for other document fields than the deployed schema's, or by another {shown(path / 'feed.py')}"
    return out


FEED_BLOCK = re.compile(r"blocked due to resource exhaustion: (?:in content cluster '\w+': )?(.*?)\. See ")


def feed(name: str, folder: pathlib.Path, advance=None, warn=print) -> float:
    """Feeds dataset name with what folder's feed.py writes, the fields its app's schema declares, through the vespa
    CLI in the container; advance(n) hears of the documents sent. Returns the seconds it took."""
    t0 = time.time()
    ds = lab.load(name)
    sent, produced = 0, lines(name, folder, ds, fields(schema(folder / "app", name), name), warn)
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        proc = subprocess.Popen(["docker", "exec", "-i", NAME, "vespa", "feed", "--target", "local", "-"],
                                stdin=subprocess.PIPE, stdout=out, stderr=err, text=True)
        try:
            for line in produced:
                proc.stdin.write(line)
                sent += 1
                if advance and sent % 200 == 0:
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
    blocked = next((m.group(1) for e in errors if (m := FEED_BLOCK.search(e))), None)
    if blocked:
        raise VespaError(f"Vespa blocks the feed: {blocked}; free some space, then lab vespa up")
    if code != 0:
        raise VespaError(f"feeding {name} failed: " + ("\n".join(cut(e)[:MAX_LINE] for e in errors[-3:])
                                                       or f"vespa feed exit code {code}"))
    try:
        ok = json.loads(stats).get("feeder.ok.count")
    except ValueError:
        ok = None
    if ok != sent:
        raise VespaError(f"feeding {name}: Vespa took {ok} of {sent} documents\n" + "\n".join(errors[-5:]))
    return time.time() - t0


def up(folder: str | pathlib.Path | None = None, progress=None, feeding=None, warn=print) -> dict:
    """Starts Vespa, deploys the search folder's app/ (default: ./search, else the lab's) and feeds every dataset whose
    feed is not in the container yet for this data, these document fields and this feed.py. progress(message) hears of
    each step, and feeding(dataset) returns the advance(n) the feed calls. Returns the deploy's answer and the feed
    times."""
    path = search_folder(folder)
    if not (lab.DATA / lab.DATA_VERSION).is_dir():
        raise VespaError("the lab's data is not here yet; run lab setup first")
    start(progress)
    answer = deploy(path, progress, warn)
    code = feed_digest(path)
    keys = {name: fingerprint(schema(path / "app", name), name, code) for name in DATASETS}
    done, seconds = fed(), {}
    todo = [name for name in DATASETS if done.get(name) != keys[name]]
    if todo and holds_documents():  # removed documents stay in Vespa's term statistics, so a feed starts empty
        if progress:
            progress(f"starting an empty {NAME} for the new feed")
        down()
        start(progress)
        answer = deploy(path, progress, warn)
        done, todo = {}, list(DATASETS)
    for name in todo:
        mark(done)
        seconds[name] = feed(name, path, feeding(name) if feeding else None, warn)
        done[name] = keys[name]
        mark(done)
    return dict(deploy=answer, fed=seconds)


def holds_documents() -> bool:
    """Whether Vespa holds any document, of any dataset."""
    status, data = http("POST", f"{QUERY_URL}/search/", json.dumps(
        {"yql": "select * from sources * where true", "hits": 0, "timeout": "10s"}).encode(), timeout=30)
    if status != 200 or data.get("root", {}).get("errors"):
        raise VespaError(f"Vespa cannot count its documents: {brief_query_error((data.get('root', {}).get('errors') or [data])[0])}")
    return data["root"].get("fields", {}).get("totalCount", 0) > 0


def down() -> bool:
    """Removes the lab-vespa container and everything fed into it; False when there was none."""
    if state() is None:
        return False
    docker("rm", "--force", NAME, timeout=120)
    return True


def status(folder: str | pathlib.Path | None = None) -> dict:
    """The container's state, memory, and per dataset its document count and why its feed is not current for the
    search folder (default: ./search, else the lab's; None when it is)."""
    s = state()
    out: dict = dict(state=s, image=IMAGE, query=QUERY_URL, config=CONFIG_URL)
    if s != "running":
        return out
    out["memory"] = docker("stats", "--no-stream", "--format", "{{.MemUsage}}", NAME).strip()
    out["serving"] = ready()
    problems, out["datasets"] = unfed(folder), {}
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
    query(qlate_bits), their centered sign bits (16 int8 per token). Questions of the data come with their tokens;
    others are encoded with lab.LATE_MODEL (on the GPU if there is one)."""
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
    return {"query(qlate_bits)": {"blocks": {str(t): b.tolist() for t, b in enumerate(bits)}},
            "query(qlate_float)": {"blocks": {str(t): v.tolist() for t, v in enumerate(vectors)}}}


_FED_CHECKED: set[str] = set()  # the datasets this process found fed for the deployed schema


def require_fed(ds_name: str) -> None:
    """Fails unless dataset ds_name's documents hold the deployed schema's fields as ./search/feed.py (else the lab's)
    writes them, so no search scores on a stale or half-done feed; checked once per process."""
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
