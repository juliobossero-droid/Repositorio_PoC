# Definición única de la PoC (para pegar en el plan de tesis)

Usar **esta misma definición** en las secciones 1.2, 4.1, Fase 2 y al inicio del Anexo A.

> **Prueba de concepto (PoC).** Verificación de viabilidad técnica del circuito completo del framework sobre un
> corpus reducido: (1) cálculo de descriptores documentales previos a la segmentación (cohesión semántica, CS, y
> variabilidad estructural jurídica, VE); (2) aplicación de una regla explícita de selección de estrategia;
> (3) segmentación con las cuatro estrategias del catálogo; (4) recuperación sobre un índice por documento; y
> (5) registro trazable de cada decisión (descriptores, regla aplicada y estrategia elegida). La PoC se considera
> lograda si el circuito se ejecuta de extremo a extremo, valida la integridad de los fragmentos (cobertura,
> offsets y límite de tokens del modelo) y produce salidas idénticas en dos ejecuciones desde cero con la misma
> configuración. **No** atribuye capacidad de validación estadística ni compara selectores: la relevancia se
> aproxima por palabras clave y la evaluación con evidencia anotada por especialistas corresponde a la Fase 2.

## Criterio de éxito (lo que reemplaza a "mejoró la cobertura de 70,9 % a 80,2 %")

| Criterio | Cómo se verifica | Dónde queda registrado |
|---|---|---|
| El circuito corre de extremo a extremo sin errores | `python -m chunklab.poc run` | `runs/<label>/REPORT.md` |
| Integridad de los chunks (cobertura, offsets, tokens reales ≤ límite) | `validate.py` en cada documento | `issues.csv` (0 errores) |
| Cada decisión es trazable | un registro por documento | `decisions.jsonl` |
| Reproducibilidad | `python -m chunklab.poc check` | salida: "REPRODUCIBLE" |
| Versiones e insumos identificados | hashes y versiones | `poc_manifest.json` |

## Esqueleto para el Anexo A (completar con los números de la corrida REAL)

- Corpus: *N* documentos (huella `…`), preguntas: *M* (sha256 `…`), modelo `intfloat/multilingual-e5-base` (revisión `…`).
- Descriptores: tabla de `descriptors.csv` (CS, CS base, VE o "no determinada").
- Decisiones: tabla de `decisions.jsonl` (regla y estrategia por documento).
- Recuperación: tabla de `retrieval_summary.csv`, con la advertencia de relevancia aproximada y *n* pequeño.
- Reproducibilidad: resultado de `check`.
- Limitaciones: las cinco de `REPORT.md`.
