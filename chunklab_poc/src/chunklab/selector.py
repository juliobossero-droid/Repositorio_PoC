"""Selector por reglas (el nivel mas simple de la 'escalera de selectores').

Es una regla EXPLICITA y AUDITABLE, fijada a priori. No esta calibrada con datos ni
se le atribuye capacidad predictiva: su unico proposito en la PoC es recorrer el
circuito completo (descriptores -> decision -> segmentacion -> recuperacion ->
registro). En la tesis, los umbrales se estimaran solo con el conjunto de
desarrollo y se compararan contra la mejor estrategia unica.

Reglas, en orden (gana la primera que se cumple):

  R1  Hay estructura juridica reconocida (encabezados 'considerando' y 'resuelvo')
      -> estrategia legal.
  R2  CS calculable y CS < cs_low (baja cohesion; umbral PROVISIONAL, null = desactivada)
      -> estrategia semantic.
  R3  Resto -> estrategia fixed (base de referencia).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SELECTOR_VERSION = "1.0"
REQUIRED_LABELS = {"considerando", "resuelvo"}

# nombre de estrategia (en la config) que usa cada regla
TARGETS = {"R1": "legal", "R2": "semantic", "R3": "fixed"}


@dataclass(frozen=True)
class Decision:
    strategy: str
    rule_id: str
    reason: str
    inputs: dict[str, Any]
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "inputs": self.inputs,
            "params": self.params,
            "selector_version": SELECTOR_VERSION,
        }


def select_strategy(desc: dict[str, Any], params: dict[str, Any]) -> Decision:
    cs_low = params.get("cs_low")
    labels = set(desc.get("heading_labels", []))
    inputs = {
        "heading_labels": sorted(labels),
        "ve_state": desc.get("ve_state"),
        "cs": desc.get("cs"),
    }
    pars = {"cs_low": cs_low}
    if REQUIRED_LABELS <= labels:
        return Decision(TARGETS["R1"], "R1", "estructura juridica reconocida (considerando + resuelvo)", inputs, pars)
    cs = desc.get("cs")
    if cs_low is not None and cs is not None and cs < float(cs_low):
        return Decision(TARGETS["R2"], "R2", f"sin estructura juridica y CS={cs:.4f} < cs_low={cs_low}", inputs, pars)
    return Decision(TARGETS["R3"], "R3", "sin estructura juridica reconocida y sin baja cohesion: estrategia base", inputs, pars)
