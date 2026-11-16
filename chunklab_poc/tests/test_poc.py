"""Pruebas de los modulos de la PoC: descriptores, evidencia, recuperacion, regla y circuito."""
import json
from pathlib import Path

import numpy as np
import pytest

from chunklab.evidence import find_evidence, fold, load_questions
from chunklab.headings import detect_headings
from chunklab.profiling import (
    cohesion_baseline, semantic_cohesion, structural_blocks, structural_variability,
)
from chunklab.retrieval import coverage_matrix, metrics, rank
from chunklab.selector import select_strategy


# ---------------------------------------------------------------- VE y CS
def test_ve_no_determinada_sin_encabezados():
    assert structural_variability([100], 0) is None
    assert structural_variability([], 0) is None


def test_ve_un_solo_bloque_no_es_cero():
    """Regresion: un encabezado al inicio deja un unico bloque; VE no debe ser 0.0."""
    text = "AUTOS Y VISTOS:\n" + "palabra " * 200
    hs, _ = detect_headings(text)
    assert len(hs) == 1
    lengths = structural_blocks(text, hs)
    assert len(lengths) == 1
    assert structural_variability(lengths, len(hs)) is None


def test_ve_bloques_iguales_es_cero_real():
    assert structural_variability([50, 50, 50], 2) == pytest.approx(0.0)


def test_ve_valor_conocido():
    # longitudes 10 y 30: media 20, desv. poblacional 10 -> CV 0.5
    assert structural_variability([10, 30], 1) == pytest.approx(0.5)


def test_cs_conocido():
    e = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert semantic_cohesion(e) == pytest.approx(0.5)  # (1 + 0) / 2


def test_cs_rechaza_no_normalizados():
    with pytest.raises(ValueError):
        semantic_cohesion(np.array([[2.0, 0.0], [0.0, 1.0]]))


def test_cs_menos_de_dos_oraciones():
    assert semantic_cohesion(np.array([[1.0, 0.0]])) is None


def test_baseline_determinista_y_dependiente_del_doc():
    rng = np.random.default_rng(0)
    e = rng.normal(size=(40, 8)); e /= np.linalg.norm(e, axis=1, keepdims=True)
    assert cohesion_baseline(e, 42, "a") == cohesion_baseline(e, 42, "a")
    assert cohesion_baseline(e, 42, "a") != cohesion_baseline(e, 42, "b")


# ---------------------------------------------------------------- evidencia
def test_fold_conserva_longitud():
    s = "Cómputo de pena ÑANDÚ  \n"
    assert len(fold(s)) == len(s)
    assert fold("Cómputo") == "computo"


def test_evidencia_sin_acentos_ni_mayusculas_y_espacios_flexibles():
    text = "El COMPUTO   de pena fue aprobado. Cómputo de pena firme."
    ev = find_evidence(text, ("cómputo de pena",))
    assert len(ev) == 2
    assert text[ev[0][0]:ev[0][1]].startswith("COMPUTO")


def test_evidencia_respeta_limite_de_palabra():
    assert find_evidence("la pena penal del penado", ("pena",)) == [(3, 7)]


def test_evidencia_fusiona_solapadas():
    text = "pena unica de prision"
    ev = find_evidence(text, ("pena", "pena unica"))
    assert ev == [(0, 10)]


def test_preguntas_csv(tmp_path):
    p = tmp_path / "q.csv"
    p.write_text("id,categoria,pregunta,keywords\n1,Pena,¿Qué pena?,pena|condena\n", encoding="utf-8")
    q = load_questions(p)
    assert q[0].keywords == ("pena", "condena")
    p.write_text("id,categoria,pregunta,keywords\n1,A,x,a\n1,A,y,b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_questions(p)


# ---------------------------------------------------------------- recuperacion
def test_cobertura_chunk_no_contiguo():
    ev = [(10, 20)]
    assert coverage_matrix([[[0, 15]]], ev)[0, 0] == False      # cubre solo parte
    assert coverage_matrix([[[0, 15], [15, 25]]], ev)[0, 0]     # spans que suman la evidencia
    assert coverage_matrix([[[8, 30]]], ev)[0, 0]


def test_metricas_ranking_conocido():
    vecs = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=float)
    q = np.array([0.0, 1.0])
    order = rank(vecs, q)
    assert list(order) == [1, 2, 0]
    covers = np.array([[False], [False], [True]])     # relevante: chunk 2 (posicion 2)
    m = metrics(order, covers, np.array([10., 20., 30.]), [1, 3])
    assert m["mrr"] == pytest.approx(0.5)
    assert m["hit@1"] == 0.0 and m["hit@3"] == 1.0
    assert m["p@3"] == pytest.approx(1 / 3)
    assert m["recall@3"] == 1.0
    assert m["tokens_top3"] == pytest.approx(20.0)


def test_empates_resueltos_por_indice():
    vecs = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert list(rank(vecs, np.array([1.0, 0.0]))) == [0, 1, 2]


# ---------------------------------------------------------------- regla
def test_regla_R1_estructura_juridica():
    d = {"heading_labels": ["vistos", "considerando", "resuelvo"], "ve_state": "calculable", "cs": 0.9}
    assert select_strategy(d, {"cs_low": 0.8}).strategy == "legal"


def test_regla_R2_baja_cohesion():
    d = {"heading_labels": [], "ve_state": "no_determinada", "cs": 0.70}
    dec = select_strategy(d, {"cs_low": 0.8})
    assert (dec.rule_id, dec.strategy) == ("R2", "semantic")
    assert select_strategy(d, {"cs_low": None}).rule_id == "R3"   # R2 desactivada


def test_regla_R3_por_defecto_y_sin_cs():
    d = {"heading_labels": ["vistos"], "ve_state": "no_determinada", "cs": None}
    assert select_strategy(d, {"cs_low": 0.8}).strategy == "fixed"


# ---------------------------------------------------------------- circuito completo
def _write_cfg(tmp_path: Path) -> Path:
    from conftest import make_text
    corpus = tmp_path / "corpus"; corpus.mkdir()
    for i in range(3):
        (corpus / f"d{i}.txt").write_text(make_text(seed=i, n_sent=50), encoding="utf-8")
    (corpus / "d3.txt").write_text(make_text(seed=9, n_sent=50, with_headings=False), encoding="utf-8")
    (tmp_path / "q.csv").write_text(
        "id,categoria,pregunta,keywords\n1,Pena,que pena,condena|pena\n2,Ley,que ley,articulo|ley\n3,Psico,informe,psicologico\n",
        encoding="utf-8",
    )
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""
run: {{label: t, seed: 1, output_dir: "{(tmp_path / 'runs').as_posix()}"}}
corpus: {{documents_dir: "{corpus.as_posix()}"}}
embedding: {{backend: hashing}}
common: {{min_tokens: 20, max_tokens: 80, safety_margin_tokens: 4}}
strategies:
  fixed: {{kind: fixed_tokens, window_tokens: 60, overlap_tokens: 10}}
  semantic: {{kind: semantic_dp, min_sentences: 2, max_sentences: 8, ideal_sentences: 4, lam: 0.4, objective: sentence_weighted}}
  louvain: {{kind: louvain_graph, tau: 0.3, gamma: 1.0, window: null, contiguous: false}}
  legal: {{kind: legal_structural, min_sentences: 2, max_sentences: 8, ideal_sentences: 4, lam: 0.4, objective: sentence_weighted, min_section_tokens: 5}}
poc: {{questions_csv: "{(tmp_path / 'q.csv').as_posix()}", ks: [1, 3], selector: {{cs_low: 0.0}}}}
""",
        encoding="utf-8",
    )
    return cfg


def test_circuito_completo_y_reproducible(tmp_path):
    from chunklab.config import load_config
    from chunklab.poc import check_reproducibility, run_poc

    cfg_path = _write_cfg(tmp_path)
    cfg = load_config(cfg_path)
    out = run_poc(cfg, cfg_path)
    for f in ("decisions.jsonl", "descriptors.csv", "retrieval_by_query.csv", "retrieval_summary.csv",
              "poc_manifest.json", "manifest.json", "REPORT.md"):
        assert (out / f).exists(), f
    dec = [json.loads(l) for l in (out / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(dec) == 4
    by_doc = {d["doc_id"]: d["decision"]["rule_id"] for d in dec}
    assert by_doc["d0"] == "R1" and by_doc["d3"] == "R3"          # d3 no tiene encabezados
    assert [d for d in dec if d["doc_id"] == "d3"][0]["descriptors"]["ve"] is None
    assert check_reproducibility(cfg, cfg_path, None) == 0
