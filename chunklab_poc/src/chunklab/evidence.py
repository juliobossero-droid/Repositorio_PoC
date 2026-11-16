"""Preguntas y evidencia de referencia.

IMPORTANTE (limitacion declarada): las preguntas del archivo CSV son GENERICAS (las
mismas para todos los documentos) y traen palabras clave, NO intervalos anotados por
especialistas. La evidencia se deriva aqui de forma automatica y determinista:

    evidencia(pregunta, documento) = union de las apariciones de cualquier palabra
    clave de la pregunta en el texto canonico del documento.

Es una relevancia APROXIMADA (etiquetado debil), util para verificar el circuito,
no para sacar conclusiones. La anotacion experta por intervalos es parte de la
Fase 2 de la tesis. Las palabras clave muy generales ('pena', 'tribunal') generan
evidencia en casi todo el documento; por eso cada fila de resultados informa
`frac_relevant` (fraccion de chunks relevantes).

Coincidencia: sin distinguir mayusculas ni diacriticos, espacios flexibles y
limites de palabra ('pena' no coincide dentro de 'penal').
"""

from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

EVIDENCE_VERSION = "1.0"


@dataclass(frozen=True)
class Question:
    qid: str
    category: str
    text: str
    keywords: tuple[str, ...]


def fold(s: str) -> str:
    """Quita diacriticos y pasa a minuscula CARACTER A CARACTER: la longitud se
    conserva y los offsets siguen siendo validos sobre el texto original."""
    out = []
    for c in s:
        base = unicodedata.normalize("NFD", c)[0]
        low = base.lower()
        out.append(low if len(low) == 1 else base)
    return "".join(out)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_questions(path: Path) -> list[Question]:
    required = {"id", "categoria", "pregunta", "keywords"}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: faltan columnas {sorted(missing)}")
        rows = list(reader)
    ids = [r["id"].strip() for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: ids de pregunta duplicados")
    out = []
    for r in rows:
        kws = tuple(k.strip() for k in r["keywords"].split("|") if k.strip())
        if not kws:
            raise ValueError(f"{path}: la pregunta {r['id']} no tiene palabras clave")
        out.append(Question(r["id"].strip(), r["categoria"].strip(), r["pregunta"].strip(), kws))
    return out


def _pattern(keyword: str) -> re.Pattern[str]:
    tokens = [re.escape(t) for t in fold(keyword).split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(tokens) + r"(?!\w)")


def find_evidence(text: str, keywords: tuple[str, ...]) -> list[tuple[int, int]]:
    """Intervalos [start, end) fusionados donde aparece alguna palabra clave."""
    folded = fold(text)
    if len(folded) != len(text):  # defensivo: nunca deberia ocurrir
        raise ValueError("fold() altero la longitud del texto")
    spans: list[tuple[int, int]] = []
    for kw in keywords:
        spans.extend((m.start(), m.end()) for m in _pattern(kw).finditer(folded))
    spans.sort()
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]
