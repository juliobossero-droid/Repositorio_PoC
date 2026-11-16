# Informe de la PoC: poc_v1_e5

## Que es y que no es este informe

Verifica la **viabilidad tecnica** del circuito completo (descriptores, regla de seleccion, segmentacion, recuperacion, registro trazable) sobre un corpus reducido. **No** valida estadisticamente ningun selector ni demuestra mejoras. La relevancia se aproxima por palabras clave (no hay evidencia anotada por especialistas) y el indice es por documento.

## Corrida

- **corpus_documentos**: `20`
- **corpus_fingerprint**: `6bd44ec6631e88690658c15d7b0a1e0f7d2db7f29006dbbc2da36e3b693d11cc`
- **preguntas**: `30`
- **preguntas_sha256**: `a39cb7c6104761ca1768f335b3aaa04975efc04dce310b2dde641dd20d198653`
- **modelo**: `intfloat/multilingual-e5-base`
- **config_hash**: `2b53681f8345cd96bc0d0e2963f7d447b0cc84ac39c0424ae64d2eb3da5ed928`
- **poc_hash**: `288dc9c8b2495c78c88f91eb6d21320c4809e90f9ad6ebf26bdccbfed6c60f17`
- **pares pregunta-documento con evidencia**: 232 de 600

## Descriptores y decision por documento

| documento | encabezados | bloques | VE | CS | CS base | regla | estrategia |
|---|---|---|---|---|---|---|---|
| Doc 0000000134 | vistos;considerando;resuelvo | 4 | 1.607 | 0.8554 | 0.8378 | R1 | legal |
| Doc 0000000141 | vistos;considerando;resuelvo | 4 | 1.563 | 0.8509 | 0.8336 | R1 | legal |
| Doc 0000000157 | vistos;considerando;resuelvo | 4 | 1.448 | 0.8499 | 0.8329 | R1 | legal |
| Doc 0000000166 | vistos;considerando;resuelvo | 4 | 1.570 | 0.8580 | 0.8388 | R1 | legal |
| Doc 0000000468 | considerando | 2 | 0.922 | 0.8580 | 0.8399 | R3 | fixed |
| Doc 0000000504 | vistos;considerando | 3 | 1.373 | 0.8654 | 0.8395 | R3 | fixed |
| Doc 0000000553 | considerando;resuelvo | 3 | 1.263 | 0.8552 | 0.8409 | R1 | legal |
| Doc 0000000600 | considerando;resuelvo | 3 | 1.235 | 0.8558 | 0.8326 | R1 | legal |
| Doc 0000000745 | considerando;resuelvo | 3 | 1.183 | 0.8631 | 0.8439 | R1 | legal |
| Doc 0000001207 | ninguno | 1 | n/d | 0.8488 | 0.8306 | R3 | fixed |
| Doc 0000001232 | ninguno | 1 | n/d | 0.8526 | n/d | R3 | fixed |
| Doc 0000001281 | considerando;resuelvo | 3 | 1.296 | 0.8497 | 0.8307 | R1 | legal |
| Doc 0000001342 | considerando;resuelvo | 3 | 1.306 | 0.8467 | 0.8381 | R1 | legal |
| Doc 0000001493 | ninguno | 1 | n/d | 0.8530 | n/d | R3 | fixed |
| Doc 0000002268 | considerando;resuelvo | 3 | 1.317 | 0.8452 | 0.8304 | R1 | legal |
| Doc 0000002716 | considerando;resuelvo | 3 | 1.235 | 0.8538 | 0.8363 | R1 | legal |
| Doc 0000002939 | considerando;resuelvo | 3 | 1.257 | 0.8474 | 0.8331 | R1 | legal |
| Doc 0000003045 | considerando;resuelvo | 3 | 1.157 | 0.8569 | 0.8420 | R1 | legal |
| Doc 0000003210 | considerando;resuelvo | 3 | 1.276 | 0.8484 | 0.8331 | R1 | legal |
| Doc 0000003272 | considerando;resuelvo | 3 | 0.540 | 0.8518 | 0.8311 | R1 | legal |

`VE = n/d` significa **no determinada** (no se pudo medir), no baja variabilidad.

## Recuperacion (media por documento, luego entre documentos)

**Preguntas todas**

| variante | n_docs | n_pares | mrr | hit@1 | hit@3 | hit@5 | p@1 | p@3 | p@5 | recall@1 | recall@3 | recall@5 | frac_relevant |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 19 | 232 | 0.691 | 0.538 | 0.854 | 0.914 | 0.538 | 0.417 | 0.347 | 0.355 | 0.696 | 0.791 | 0.215 |
| legal | 19 | 232 | 0.700 | 0.569 | 0.812 | 0.896 | 0.569 | 0.371 | 0.289 | 0.396 | 0.673 | 0.792 | 0.159 |
| louvain | 19 | 232 | 0.689 | 0.536 | 0.808 | 0.905 | 0.536 | 0.373 | 0.289 | 0.378 | 0.671 | 0.780 | 0.162 |
| semantic | 19 | 232 | 0.720 | 0.561 | 0.853 | 0.925 | 0.561 | 0.389 | 0.300 | 0.378 | 0.717 | 0.816 | 0.169 |
| SELECTOR_REGLAS | 19 | 232 | 0.696 | 0.562 | 0.805 | 0.884 | 0.562 | 0.384 | 0.303 | 0.387 | 0.663 | 0.779 | 0.177 |

**Preguntas discriminativas** (ninguna estrategia con mas de 50% de chunks relevantes; mismos pares para todas)

| variante | n_docs | n_pares | mrr | hit@1 | hit@3 | hit@5 | p@1 | p@3 | p@5 | recall@1 | recall@3 | recall@5 | frac_relevant |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 19 | 227 | 0.689 | 0.534 | 0.853 | 0.913 | 0.534 | 0.401 | 0.327 | 0.369 | 0.702 | 0.793 | 0.195 |
| legal | 19 | 227 | 0.700 | 0.570 | 0.811 | 0.895 | 0.570 | 0.360 | 0.279 | 0.408 | 0.676 | 0.794 | 0.148 |
| louvain | 19 | 227 | 0.690 | 0.539 | 0.807 | 0.904 | 0.539 | 0.363 | 0.279 | 0.390 | 0.673 | 0.783 | 0.152 |
| semantic | 19 | 227 | 0.719 | 0.560 | 0.852 | 0.925 | 0.560 | 0.379 | 0.289 | 0.390 | 0.720 | 0.820 | 0.157 |
| SELECTOR_REGLAS | 19 | 227 | 0.696 | 0.562 | 0.803 | 0.884 | 0.562 | 0.369 | 0.284 | 0.402 | 0.670 | 0.782 | 0.158 |

## Limitaciones declaradas

- Pocos documentos: no hay inferencia estadistica ni intervalos de confianza.
- Relevancia por palabras clave (etiquetado debil); palabras generales generan evidencia en gran parte del texto (ver `frac_relevant`).
- Indice por documento, sin documentos distractores ni presupuesto de recuperacion normalizado.
- La regla de seleccion esta fijada a priori y no esta calibrada; el umbral `cs_low` es provisional.
- CS y VE dependen del modelo de embeddings y del detector de encabezados (version registrada en el manifiesto).
