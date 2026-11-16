# chunklab · PoC de selección adaptativa y trazable de estrategias de chunking

Prueba de concepto (PoC) del plan de tesis *"Diseño y evaluación de un framework para la selección
adaptativa y trazable de estrategias de segmentación documental en sistemas RAG"* (UNLaM).

## Qué hace (y qué no)

Recorre **una vez el circuito completo** sobre un corpus reducido:

```
documento → descriptores (CS, VE) → regla de selección → segmentación (4 estrategias)
          → recuperación → registro trazable de la decisión
```

Verifica **viabilidad técnica y reproducibilidad**. **No** valida estadísticamente ningún selector ni
demuestra mejoras: pocos documentos, relevancia aproximada por palabras clave, índice por documento.

## Ejecutar (Windows / PowerShell)

```powershell
cd chunklab_poc
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pip freeze > requirements.lock          # congelar el entorno exacto

python -m pytest -q                      # 1) tests (deben pasar)
python -m chunklab.poc run   --config config/poc_smoke.yaml   # 2) simulacro sin modelo (SIN validez científica)
python -m chunklab.poc run   --config config/poc.yaml         # 3) corrida REAL con e5-base (descarga el modelo)
python -m chunklab.poc check --config config/poc.yaml         # 4) reproducibilidad: dos corridas desde cero, salidas idénticas
```

Los documentos `.txt` van en `data/corpus/` (o cambiar `corpus.documents_dir` en la config).
Las preguntas están en `data/preguntas_muestra_30.csv` (30) y `data/preguntas.csv` (100).

## Qué produce (`runs/<label>/`)

| Archivo | Contenido |
|---|---|
| `REPORT.md` | Informe legible: descriptores, decisión por documento, métricas y limitaciones |
| `decisions.jsonl` | **Registro trazable**: por documento, descriptores → regla aplicada → estrategia y motivo |
| `descriptors.csv` | CS, CS base, VE (o `no_determinada`), encabezados, bloques |
| `retrieval_by_query.csv` | Una fila por documento × pregunta × estrategia (P@k, R@k, Hit@k, MRR, `frac_relevant`) |
| `retrieval_summary.csv` | Medias por documento y luego entre documentos; incluye la fila `SELECTOR_REGLAS` |
| `chunks/*.jsonl`, `shards/` | Chunks con offsets (confidencial: contiene texto) |
| `manifest.json`, `poc_manifest.json` | Hashes (config, corpus, preguntas), versiones de código y librerías, modelo |

## Reproducibilidad: cómo está resuelta

- Semillas fijas; ranking con desempate determinista; salidas redondeadas a 6 decimales y ordenadas.
- Hashes en el manifiesto: **configuración**, **corpus** (huella por documento), **archivo de preguntas** y
  **parámetros de la PoC**. Si algo cambia, el hash cambia; un shard viejo con otra configuración aborta la corrida.
- Versiones registradas: detector de encabezados, segmentador de oraciones, descriptores, regla, evidencia, recuperación.
- Lectura canónica única del corpus (`io.canonicalize`: UTF-8, `\n`, NFC): todos los offsets refieren a ese texto.
- `chunklab.poc check` corre la PoC dos veces desde cero (sin caché) y compara los hash de las salidas.
- **Pendiente de tu lado antes de la corrida real:** fijar `embedding.revision` (commit del modelo en HuggingFace)
  y guardar `requirements.lock`. El manifiesto avisa si la revisión no está fijada.

## Decisiones de diseño (podés revertirlas)

1. **Prefijos e5**: oraciones `query: ` (similitud simétrica: CS y segmentación semántica), chunks `passage: `,
   preguntas `query: `. Antes los scripts sueltos usaban `passage: ` para CS; los valores de CS **cambian** y hay que
   usar los nuevos en el Anexo.
2. **Un único detector de encabezados (v1.0)**, el del repo. La extensión a apartados romanos (V2) agregaba un tipo
   de encabezado (`apartado`) que la estrategia legal no contempla, y no estaba validada. Queda como línea futura.
3. **VE nunca vale 0 por defecto**: sin encabezados o con un solo bloque es `no_determinada`. (Error del código
   anterior: un documento con un único encabezado inicial quedaba con VE = 0,0 "calculable".)
4. **Relevancia por palabras clave** (sin acentos/mayúsculas, espacios flexibles, límites de palabra). Un chunk es
   relevante si contiene completa alguna aparición. Etiquetado débil, declarado como limitación.
5. **Índice en memoria por documento** (numpy), sin ChromaDB: más simple y determinista. El índice multidocumento con
   distractores corresponde a la Fase 2 de la tesis.
6. **Regla de selección** (R1 estructura jurídica → legal; R2 CS < `cs_low` → semantic; R3 → fixed). `cs_low = 0.80`
   es un valor **provisional** fijado a priori, no calibrado.

## Cambios respecto del repositorio original

Nuevos: `profiling.py`, `selector.py`, `evidence.py`, `retrieval.py`, `poc.py`, `tests/test_poc.py`, `config/`.
Modificados: `pipeline.py` (agrega `descriptors` al shard; esquema 2), `config.py` (la versión de `profiling` entra
en el hash), `tests/test_token_drift.py` (se omite si no hay modelo o documento; ya no usa `C:/LC`).
No se tocó `strategies/`.

## Qué falta para la tesis (fuera de esta PoC)

Anotación experta por intervalos (≥ 2 especialistas), índice multidocumento con distractores, particiones
desarrollo/evaluación, estimación de umbrales solo con desarrollo, clasificadores y agentes, evaluación de transferencia.
