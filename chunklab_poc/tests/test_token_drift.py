
"""Prueba de regresión de TOKEN_DRIFT con el modelo real."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from chunklab.config import CommonConfig
from chunklab.core import Document
from chunklab.io import read_text_file, canonicalize
from chunklab.strategies import DocView, KINDS, finalize
from chunklab.tokenization import HFTokenizerAdapter, TokenIndex


@pytest.mark.parametrize("overlap", [0, 50])
def test_fixed_real_token_limit(overlap):
    """La estrategia fija debe respetar 256 tokens reales."""

    # Requiere el modelo real (descarga desde HuggingFace) y un documento local.
    # Se omite si no estan disponibles. Ruta configurable: CHUNKLAB_REAL_DOC.
    pytest.importorskip("transformers")
    default = Path(__file__).resolve().parents[1] / "data" / "corpus" / "Doc_0000000003.txt"
    path = Path(os.environ.get("CHUNKLAB_REAL_DOC", default))
    if not path.exists():
        pytest.skip(f"No hay documento real en {path}")

    # Leer y normalizar el documento.
    raw, encoding = read_text_file(path)
    text = canonicalize(raw)

    doc = Document(
        doc_id=path.stem,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        source_path=str(path),
        encoding=encoding,
    )

    # Cargar el tokenizador real.
    tokenizer = HFTokenizerAdapter.from_pretrained(
        "intfloat/multilingual-e5-base"
    )

    token_index = TokenIndex(tokenizer.offsets(doc.text))

    common = CommonConfig(
        min_tokens=64,
        max_tokens=256,
        safety_margin_tokens=8,
    )

    # Preparar la vista del documento.
    view = DocView(
        doc=doc,
        token_index=token_index,
        sentences=[],
        sentence_tokens=np.array([], dtype=int),
        headings=[],
        strict_headings=False,
    )

    # Ejecutar la estrategia fija.
    strategy = KINDS["fixed_tokens"](
        "fixed",
        {"window_tokens": 256, "overlap_tokens": overlap},
        common,
        seed=42,
    )

    chunks = finalize(view, strategy.chunk(view), "fixed")

    # Verificar los tokens reales de cada fragmento.
    oversized = [
        (c.index, tokenizer.count(c.text))
        for c in chunks
        if tokenizer.count(c.text) > 256
    ]

    assert not oversized, (
        f"TOKEN_DRIFT con overlap={overlap}: {oversized}"
    )
