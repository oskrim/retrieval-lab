"""Trains the LightGBM model of the ltr rank profile (vespa/app/models/ltr.json) on the hybrid profile's match features
of the fiqa test questions' top 100, as notebook 13 trains on the lexical features. Needs a running, fed Vespa.

  python tools/vespa_ltr.py [vespa/app]
"""

import importlib.util
import json
import pathlib
import sys

import lightgbm as lgb
import numpy as np

import lab
import labvespa

FEATURES = ("bm25(body)", "bm25(title)", "phrase_share", "dense", "cosine")  # match-features of the hybrid profile
GBM = dict(objective="lambdarank", learning_rate=0.05, num_leaves=7, deterministic=True, force_row_wise=True, verbose=-1)


def main(app: pathlib.Path) -> None:
    spec = importlib.util.spec_from_file_location("vespa_search", lab.HERE / "templates" / "vespa" / "search.py")
    search = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(search)
    ds = lab.load("fiqa", "test")
    mean, _, known = lab.dense_parts(ds)
    X, y, groups = [], [], []
    for qid in ds.qids:
        question = ds.queries[qid]
        terms, pairs = search.words(question)
        q = known[question]
        body = {"yql": search.yql(ds, question, terms, pairs), "ranking": "hybrid", "hits": 100, "q": " ".join(terms),
                "model.restrict": ds.name, "timeout": "10s", "input.query(q)": q.tolist(), "input.query(qc)": (q - mean).tolist(),
                "input.query(qbits)": lab.pack_signs(q - mean >= 0).view(np.int8).tolist(), "input.query(pairs)": len(pairs)}
        status, data = labvespa.http("POST", f"{labvespa.QUERY_URL}/search/", json.dumps(body).encode())
        hits = data.get("root", {}).get("children", [])
        if status != 200 or not hits:
            raise SystemExit(f"Vespa answered {status}: {data.get('root', data)}")
        X += [[h["fields"]["matchfeatures"][f] for f in FEATURES] for h in hits]
        y += [ds.qrels[qid].get(h["fields"]["id"], 0) for h in hits]
        groups.append(len(hits))
    model = lgb.train(GBM, lgb.Dataset(np.array(X), np.array(y), group=groups, feature_name=list(FEATURES)), 200)
    out = app / "models" / "ltr.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(model.dump_model(), separators=(",", ":")))
    print(f"wrote {out} ({len(groups)} questions, {len(y)} documents); next: lab vespa deploy")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else lab.HERE / "vespa" / "app"))
