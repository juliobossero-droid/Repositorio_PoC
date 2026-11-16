"""Prueba de concepto (PoC): recorre el circuito completo sobre un corpus reducido.

    documento -> descriptores (CS, VE) -> regla de seleccion -> segmentacion
              -> recuperacion -> registro trazable

Uso:
    python -m chunklab.poc run   --config config/poc.yaml [--force] [--limit N]
    python -m chunklab.poc check --config config/poc.yaml     # doble corrida y comparacion

La PoC verifica VIABILIDAD TECNICA. No atribuye capacidad de validacion estadistica
ni compara selectores: con pocos documentos y relevancia por palabras clave, los
numeros solo muestran que el circuito funciona y que es reproducible.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import logging
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .config import Config, ConfigError, load_config
from .evidence import EVIDENCE_VERSION, Question, find_evidence, load_questions, sha256_file
from .export import export_run
from .headings import HEADINGS_VERSION
from .io import atomic_write_text, corpus_fingerprint, dump_json, load_documents
from .pipeline import SCHEMA, build_runtime, run_corpus, shard_path
from .profiling import PROFILING_VERSION
from .retrieval import RETRIEVAL_VERSION, PrefixedEncoder, coverage_matrix, metrics, rank
from .selector import SELECTOR_VERSION, TARGETS, select_strategy
from .sentences import SEGMENTER_VERSION

LOG = logging.getLogger("chunklab.poc")
DISC_THRESHOLD = 0.5  # una pregunta es 'discriminativa' si <= 50 % de los chunks son relevantes


# ------------------------------------------------------------------ utilidades
def _r(x: Any, nd: int = 6) -> Any:
    """Redondeo para que las salidas sean estables entre plataformas."""
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    return x


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def _libs() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in ("numpy", "PyYAML", "networkx", "transformers", "sentence-transformers", "torch", "tqdm"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def poc_section(cfg: Config, config_path: Path) -> dict[str, Any]:
    p = dict(cfg.raw.get("poc") or {})
    if "questions_csv" not in p:
        raise ConfigError("Falta 'poc.questions_csv' en la configuracion")
    q = Path(p["questions_csv"])
    p["questions_csv"] = str(q if q.is_absolute() else (config_path.parent / q).resolve())
    p.setdefault("ks", [1, 3, 5])
    p.setdefault("selector", {"cs_low": None})
    p.setdefault("chunk_prefix", "passage: ")
    p.setdefault("query_prefix", "query: ")
    p["ks"] = sorted({int(k) for k in p["ks"]})
    missing = [t for t in TARGETS.values() if t not in {s.name for s in cfg.strategies}]
    if missing:
        raise ConfigError(f"El selector necesita las estrategias {missing} habilitadas en 'strategies'")
    return p


# ------------------------------------------------------------------ evaluacion
METRIC_COLS = ["mrr"]


def evaluate(cfg: Config, poc: dict[str, Any], docs, questions: list[Question], rt, out: Path):
    ks = poc["ks"]
    qenc = PrefixedEncoder(rt.embedder, poc["query_prefix"])
    cenc = PrefixedEncoder(rt.embedder, poc["chunk_prefix"])
    qvecs = qenc.encode([q.text for q in questions])
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    desc_rows: list[dict[str, Any]] = []
    pair_counts = {"pares_posibles": 0, "pares_con_evidencia": 0}
    for doc in docs:
        shard = json.loads(shard_path(out, doc.doc_id).read_text(encoding="utf-8"))
        desc = shard["descriptors"]
        dec = select_strategy(desc, poc["selector"])
        decisions.append(
            _r(
                {
                    "doc_id": doc.doc_id,
                    "doc_sha256": doc.sha256,
                    "config_hash": shard["config_hash"],
                    "descriptors": desc,
                    "decision": dec.to_dict(),
                }
            )
        )
        desc_rows.append(
            _r(
                {
                    "doc_id": doc.doc_id,
                    **{k: (";".join(v) if isinstance(v, list) else v) for k, v in desc.items()},
                    "regla": dec.rule_id,
                    "estrategia_elegida": dec.strategy,
                }
            )
        )
        evidence = {q.qid: find_evidence(doc.text, q.keywords) for q in questions}
        pair_counts["pares_posibles"] += len(questions)
        pair_counts["pares_con_evidencia"] += sum(1 for v in evidence.values() if v)
        for sname, chunks in shard["strategies"].items():
            cvecs = cenc.encode([c["text"] for c in chunks])
            spans = [c["spans"] for c in chunks]
            toks = np.array([c["n_tokens"] for c in chunks], dtype=np.float64)
            for qi, q in enumerate(questions):
                ev = evidence[q.qid]
                if not ev:
                    continue
                m = metrics(rank(cvecs, qvecs[qi]), coverage_matrix(spans, ev), toks, ks)
                rows.append(
                    _r(
                        {
                            "doc_id": doc.doc_id,
                            "query_id": q.qid,
                            "categoria": q.category,
                            "strategy": sname,
                            "seleccionada": int(sname == dec.strategy),
                            "n_chunks": len(chunks),
                            "n_evidencias": len(ev),
                            **m,
                        }
                    )
                )
    return rows, decisions, desc_rows, pair_counts


def summarize(rows: list[dict[str, Any]], ks: list[int]) -> list[dict[str, Any]]:
    """Media por documento y luego entre documentos (el documento es la unidad).

    'discriminativas': pares (documento, pregunta) en los que NINGUNA estrategia tiene
    mas de DISC_THRESHOLD de chunks relevantes. El conjunto de pares es el mismo para
    todas las variantes, de modo que la comparacion es pareada."""
    cols = ["mrr"] + [f"{m}@{k}" for k in ks for m in ("hit", "p", "recall")] + [f"tokens_top{max(ks)}", "frac_relevant"]
    worst: dict[tuple[str, str], float] = defaultdict(float)
    for r in rows:
        key = (r["doc_id"], r["query_id"])
        worst[key] = max(worst[key], r["frac_relevant"])
    out: list[dict[str, Any]] = []
    subsets = (
        ("todas", lambda r: True),
        ("discriminativas", lambda r: worst[(r["doc_id"], r["query_id"])] <= DISC_THRESHOLD),
    )
    for subset, keep in subsets:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            if not keep(r):
                continue
            groups[r["strategy"]].append(r)
            if r["seleccionada"]:
                groups["SELECTOR_REGLAS"].append(r)
        for name in sorted(groups, key=lambda n: (n == "SELECTOR_REGLAS", n)):
            by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in groups[name]:
                by_doc[r["doc_id"]].append(r)
            row: dict[str, Any] = {"subconjunto": subset, "variante": name, "n_docs": len(by_doc), "n_pares": len(groups[name])}
            for c in cols:
                row[c] = float(np.mean([np.mean([r[c] for r in rs]) for rs in by_doc.values()]))
            out.append(_r(row))
    return out


# ------------------------------------------------------------------ reporte
def _fmt(x: Any, nd: int = 3) -> str:
    return "n/d" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def write_report(out: Path, cfg: Config, poc: dict[str, Any], man: dict[str, Any], desc_rows, summary, pairs) -> None:
    sim = cfg.embedding.backend == "hashing"
    L: list[str] = []
    L.append(f"# Informe de la PoC: {cfg.label}\n")
    if sim:
        L.append("> **SIMULACRO**: se uso el backend `hashing`. Los resultados NO tienen validez cientifica; solo prueban que el circuito corre.\n")
    L.append("## Que es y que no es este informe\n")
    L.append(
        "Verifica la **viabilidad tecnica** del circuito completo (descriptores, regla de seleccion, segmentacion, recuperacion, registro trazable) "
        "sobre un corpus reducido. **No** valida estadisticamente ningun selector ni demuestra mejoras. La relevancia se aproxima por palabras clave "
        "(no hay evidencia anotada por especialistas) y el indice es por documento.\n"
    )
    L.append("## Corrida\n")
    for k in ("corpus_documentos", "corpus_fingerprint", "preguntas", "preguntas_sha256", "modelo", "config_hash", "poc_hash"):
        L.append(f"- **{k}**: `{man[k]}`")
    L.append(f"- **pares pregunta-documento con evidencia**: {pairs['pares_con_evidencia']} de {pairs['pares_posibles']}\n")
    L.append("## Descriptores y decision por documento\n")
    L.append("| documento | encabezados | bloques | VE | CS | CS base | regla | estrategia |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in desc_rows:
        L.append(
            f"| {r['doc_id']} | {r['heading_labels'] or 'ninguno'} | {r['n_blocks']} | {_fmt(r['ve'])} | {_fmt(r['cs'], 4)} | "
            f"{_fmt(r['cs_baseline'], 4)} | {r['regla']} | {r['estrategia_elegida']} |"
        )
    L.append("\n`VE = n/d` significa **no determinada** (no se pudo medir), no baja variabilidad.\n")
    L.append("## Recuperacion (media por documento, luego entre documentos)\n")
    ks = poc["ks"]
    head = ["variante", "n_docs", "n_pares", "mrr"] + [f"hit@{k}" for k in ks] + [f"p@{k}" for k in ks] + [f"recall@{k}" for k in ks] + ["frac_relevant"]
    for subset in ("todas", "discriminativas"):
        L.append(f"**Preguntas {subset}**" + (f" (ninguna estrategia con mas de {DISC_THRESHOLD:.0%} de chunks relevantes; mismos pares para todas)" if subset == "discriminativas" else "") + "\n")
        L.append("| " + " | ".join(head) + " |")
        L.append("|" + "---|" * len(head))
        for r in summary:
            if r["subconjunto"] == subset:
                L.append("| " + " | ".join(_fmt(r[h], 3) if h not in ("variante", "n_docs", "n_pares") else str(r[h]) for h in head) + " |")
        L.append("")
    L.append("## Limitaciones declaradas\n")
    for t in (
        "Pocos documentos: no hay inferencia estadistica ni intervalos de confianza.",
        "Relevancia por palabras clave (etiquetado debil); palabras generales generan evidencia en gran parte del texto (ver `frac_relevant`).",
        "Indice por documento, sin documentos distractores ni presupuesto de recuperacion normalizado.",
        "La regla de seleccion esta fijada a priori y no esta calibrada; el umbral `cs_low` es provisional.",
        "CS y VE dependen del modelo de embeddings y del detector de encabezados (version registrada en el manifiesto).",
    ):
        L.append(f"- {t}")
    atomic_write_text(out / "REPORT.md", "\n".join(L) + "\n")


# ------------------------------------------------------------------ corrida
def run_poc(cfg: Config, config_path: Path, *, force: bool = False, limit: int | None = None) -> Path:
    poc = poc_section(cfg, config_path)
    docs = load_documents(cfg.corpus)
    if limit:
        docs = docs[:limit]
    qpath = Path(poc["questions_csv"])
    questions = load_questions(qpath)
    LOG.info("Documentos: %d | preguntas: %d | config %s", len(docs), len(questions), cfg.fingerprint()[:12])
    rt = build_runtime(cfg)
    out = Path(cfg.output_dir) / cfg.label
    summ = run_corpus(docs, rt, out, force=force)
    if summ["failed"]:
        raise RuntimeError(f"{summ['failed']} documentos fallaron; ver {out / 'errors.jsonl'}")
    export_run(cfg, out, docs, extra={"run_summary": summ, "embedder": rt.embedder.fingerprint(), "hard_limit": rt.hard_limit})
    rows, decisions, desc_rows, pairs = evaluate(cfg, poc, docs, questions, rt, out)
    summary = summarize(rows, poc["ks"])
    atomic_write_text(out / "decisions.jsonl", "\n".join(json.dumps(d, ensure_ascii=False, sort_keys=True) for d in decisions) + "\n")
    _write_csv(out / "descriptors.csv", desc_rows)
    _write_csv(out / "retrieval_by_query.csv", rows)
    _write_csv(out / "retrieval_summary.csv", summary)
    poc_hash = hashlib.sha256(
        json.dumps(
            {
                "poc": {k: v for k, v in poc.items() if k != "questions_csv"},
                "questions_sha256": sha256_file(qpath),
                "versions": [RETRIEVAL_VERSION, EVIDENCE_VERSION, SELECTOR_VERSION],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    man = {
        "chunklab": __version__,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": _libs(),
        "simulacro_sin_validez_cientifica": cfg.embedding.backend == "hashing",
        "modelo": cfg.embedding.model if cfg.embedding.backend != "hashing" else "hashing",
        "modelo_revision_fijada": cfg.embedding.revision is not None,
        "embedder": rt.embedder.fingerprint(),
        "config_hash": cfg.fingerprint(),
        "poc_hash": poc_hash,
        "poc": poc,
        "corpus_documentos": len(docs),
        "corpus_fingerprint": corpus_fingerprint(docs),
        "preguntas": len(questions),
        "preguntas_sha256": sha256_file(qpath),
        "versiones": {
            "shard_schema": SCHEMA,
            "headings": HEADINGS_VERSION,
            "segmentador_oraciones": SEGMENTER_VERSION,
            "profiling": PROFILING_VERSION,
            "selector": SELECTOR_VERSION,
            "evidencia": EVIDENCE_VERSION,
            "recuperacion": RETRIEVAL_VERSION,
        },
        **pairs,
    }
    dump_json(out / "poc_manifest.json", man)
    write_report(out, cfg, poc, man, desc_rows, summary, pairs)
    print(f"PoC lista: {len(docs)} documentos, {len(rows)} evaluaciones. Resultados en: {out}")
    return out


REPRO_FILES = ["decisions.jsonl", "descriptors.csv", "retrieval_by_query.csv", "retrieval_summary.csv"]


def check_reproducibility(cfg: Config, config_path: Path, limit: int | None) -> int:
    """Corre la PoC dos veces desde cero (sin cache ni shards) y compara las salidas."""
    outs = []
    for tag in ("chkA", "chkB"):
        label = f"{cfg.label}_{tag}"
        out_dir = Path(cfg.output_dir) / label
        c = dataclasses.replace(cfg, label=label, cache_dir=str(out_dir / "_cache"))
        outs.append(run_poc(c, config_path, force=True, limit=limit))

    def digests(d: Path) -> dict[str, str]:
        files = REPRO_FILES + [f"chunks/{p.name}" for p in sorted((d / "chunks").glob("*.jsonl"))]
        return {f: sha256_file(d / f) for f in files}

    a, b = digests(outs[0]), digests(outs[1])
    ok = a == b
    for f in a:
        print(f"  {'IGUAL    ' if a[f] == b.get(f) else 'DIFERENTE'} {f}")
    print("REPRODUCIBLE: las dos corridas producen salidas identicas." if ok else "NO REPRODUCIBLE: hay diferencias.")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="chunklab.poc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "check"):
        s = sub.add_parser(name)
        s.add_argument("--config", default="config/poc.yaml")
        s.add_argument("--limit", type=int)
        if name == "run":
            s.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        cfg_path = Path(args.config).resolve()
        cfg = load_config(cfg_path)
        if args.cmd == "run":
            run_poc(cfg, cfg_path, force=args.force, limit=args.limit)
            return 0
        return check_reproducibility(cfg, cfg_path, args.limit)
    except ConfigError as exc:
        print(f"Error de configuracion: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
