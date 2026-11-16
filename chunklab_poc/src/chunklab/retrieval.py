"""Recuperacion y metricas.

Indice: en memoria, POR DOCUMENTO (la PoC no usa indice multidocumento con
distractores; eso corresponde al protocolo de la Fase 2). Similitud: producto
punto de vectores normalizados (= coseno). Empates: por indice de chunk, de modo que
el ranking es determinista.

Relevancia de un chunk: contiene COMPLETA al menos una evidencia (cuenta la
cobertura sumada de todos sus spans, porque en Louvain un chunk puede no ser
contiguo).

Metricas (k en `ks`):
  hit@k        1 si hay algun chunk relevante entre los k primeros.
  p@k          relevantes entre los k primeros / min(k, n_chunks).
  recall@k     evidencias completamente contenidas en algun chunk de los k primeros
               / total de evidencias.
  mrr          1 / posicion del primer relevante en el ranking completo (0 si ninguno).
  frac_relevant  fraccion de chunks relevantes (diagnostico de preguntas triviales).
  tokens_topk  tokens medios de los chunks recuperados en top-k (costo de contexto).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

RETRIEVAL_VERSION = "1.0"


class PrefixedEncoder:
    """Misma red que `base`, con otro prefijo (e5: 'query: ' / 'passage: ').
    Con el backend de simulacro (hashing) el prefijo no se usa."""

    def __init__(self, base: Any, prefix: str):
        self.base = base
        self.prefix = prefix

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not len(texts):
            return np.empty((0, self.base.dim), dtype=np.float32)
        if getattr(self.base, "backend", "") == "sentence-transformers":
            arr = self.base.model.encode(
                [self.prefix + t for t in texts],
                batch_size=self.base.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(arr, dtype=np.float32)
        return self.base.encode(list(texts))

    def fingerprint(self) -> str:
        return f"{self.base.fingerprint()}|retrieval_prefix={self.prefix!r}"


def coverage_matrix(chunk_spans: Sequence[Sequence[Sequence[int]]], evidence: Sequence[tuple[int, int]]) -> np.ndarray:
    """Matriz booleana (n_chunks, n_evidencias): el chunk contiene completa la evidencia."""
    es = np.array([e[0] for e in evidence], dtype=np.int64)
    ee = np.array([e[1] for e in evidence], dtype=np.int64)
    need = ee - es
    out = np.zeros((len(chunk_spans), len(evidence)), dtype=bool)
    for i, spans in enumerate(chunk_spans):
        ov = np.zeros(len(evidence), dtype=np.int64)
        for s, e in spans:
            ov += np.maximum(0, np.minimum(e, ee) - np.maximum(s, es))
        out[i] = ov >= need
    return out


def rank(chunk_vecs: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    scores = chunk_vecs @ query_vec
    return np.lexsort((np.arange(len(scores)), -scores))


def metrics(order: np.ndarray, covers: np.ndarray, tokens: np.ndarray, ks: Sequence[int]) -> dict[str, float]:
    n = len(order)
    relevant = covers.any(axis=1)
    rel_sorted = relevant[order]
    out: dict[str, float] = {"frac_relevant": float(relevant.mean())}
    pos = np.flatnonzero(rel_sorted)
    out["mrr"] = float(1.0 / (pos[0] + 1)) if len(pos) else 0.0
    n_ev = covers.shape[1]
    for k in ks:
        top = order[: min(k, n)]
        out[f"hit@{k}"] = float(relevant[top].any())
        out[f"p@{k}"] = float(relevant[top].sum() / len(top))
        out[f"recall@{k}"] = float(covers[top].any(axis=0).sum() / n_ev)
        out[f"tokens_top{k}"] = float(tokens[top].mean())
    return out
