"""Descriptores documentales calculables ANTES de segmentar: CS y VE.

Definiciones (las mismas que se declaran en el Anexo A del plan de tesis):

* CS (cohesion semantica): media del coseno entre los embeddings de oraciones
  consecutivas. Las oraciones son las unidades del pipeline (oraciones
  atomizadas a <= max_tokens). Se acompana de `cs_baseline`: media del coseno
  entre pares de oraciones NO cercanas (|i-j| >= 5), para saber si el CS observado
  supera lo esperable por la mera homogeneidad tematica del corpus o del modelo.
* VE (variabilidad estructural juridica): coeficiente de variacion poblacional
  (desv. estandar / media) de las longitudes en caracteres de los bloques
  delimitados por los encabezados detectados (incluye el texto previo al primer
  encabezado y el posterior al ultimo).

VE es 'no_determinada' (None) cuando no hay encabezados o hay menos de 2 bloques.
NUNCA se rellena con 0: un 0 significaria "bloques perfectamente uniformes", y eso
es distinto de "no se pudo medir".
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np

PROFILING_VERSION = "1.0"
BASELINE_MIN_GAP = 5
BASELINE_PAIRS = 500


def structural_blocks(text: str, headings: list) -> list[int]:
    """Longitudes (en caracteres, sin blancos en los extremos) de los bloques."""
    cuts = [0] + sorted({h.start for h in headings if 0 < h.start < len(text)}) + [len(text)]
    return [len(text[a:b].strip()) for a, b in zip(cuts, cuts[1:]) if text[a:b].strip()]


def structural_variability(lengths: list[int], n_headings: int) -> float | None:
    if n_headings == 0 or len(lengths) < 2:
        return None
    arr = np.asarray(lengths, dtype=np.float64)
    mean = float(arr.mean())
    if mean <= 0:
        return None
    return float(arr.std(ddof=0) / mean)


def _check_matrix(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError("Se esperaba una matriz (n, dim) de embeddings")
    if len(emb) and not np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3):
        raise ValueError("Los embeddings deben estar normalizados (L2)")
    return emb


def semantic_cohesion(emb: np.ndarray) -> float | None:
    emb = _check_matrix(emb)
    if len(emb) < 2:
        return None
    return float(np.mean(np.sum(emb[:-1] * emb[1:], axis=1)))


def cohesion_baseline(emb: np.ndarray, seed: int, doc_id: str) -> float | None:
    """Coseno medio entre pares de oraciones no cercanas (determinista por seed+doc)."""
    emb = _check_matrix(emb)
    n = len(emb)
    if n < BASELINE_MIN_GAP + 2:
        return None
    rng = np.random.default_rng([int(seed), zlib.crc32(doc_id.encode("utf-8"))])
    i = rng.integers(0, n, BASELINE_PAIRS * 4)
    j = rng.integers(0, n, BASELINE_PAIRS * 4)
    keep = np.abs(i - j) >= BASELINE_MIN_GAP
    i, j = i[keep][:BASELINE_PAIRS], j[keep][:BASELINE_PAIRS]
    if len(i) == 0:
        return None
    return float(np.mean(np.sum(emb[i] * emb[j], axis=1)))


def compute_descriptors(view: Any, seed: int) -> dict[str, Any]:
    """Descriptores de un documento a partir de su DocView (ver strategies.base)."""
    text = view.doc.text
    headings = view.headings
    lengths = structural_blocks(text, headings)
    ve = structural_variability(lengths, len(headings))
    emb = view.embeddings()
    cs = semantic_cohesion(emb)
    base = cohesion_baseline(emb, seed, view.doc.doc_id)
    return {
        "profiling_version": PROFILING_VERSION,
        "n_chars": len(text),
        "n_sentences": len(view.sentences),
        "n_headings": len(headings),
        "heading_labels": [h.label for h in headings],
        "n_blocks": len(lengths),
        "ve": ve,
        "ve_state": "calculable" if ve is not None else "no_determinada",
        "cs": cs,
        "cs_state": "calculable" if cs is not None else "no_determinada",
        "cs_baseline": base,
        "cs_minus_baseline": (cs - base) if (cs is not None and base is not None) else None,
    }
