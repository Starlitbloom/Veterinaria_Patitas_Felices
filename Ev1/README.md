# Veterinaria_Patitas_Felices
Prototipo de asistente virtual basado en LLM + RAG (Retrieval-Augmented Generation) para la Veterinaria Patitas Felices (Rancagua). Responde consultas frecuentes de dueños de mascotas (horarios, agendamiento, especies atendidas, vacunas, historial clínico) usando únicamente información recuperada de fuentes internas y externas de la clínica, y deriva a un veterinario humano cualquier consulta clínica.
 
Este repositorio corresponde a la Evaluación Parcial N°1 de la asignatura Ingeniería de Soluciones con IA (ISY0101).
 
## Contenido del repositorio
 
- `asistente_rag.py` — Script principal. Implementa el pipeline RAG completo: ingesta de fuentes, recuperación (retriever), ensamblado de prompt, generación de respuesta vía LLM y registro de logs.
- `arquitectura.png` — Diagrama de la arquitectura de la solución.
- `evidencia/evidencia_pruebas.txt` — Evidencia de una ejecución de prueba del script, con las consultas usadas y las respuestas obtenidas.
- `evidencia/logs_consultas.jsonl` — Log de trazabilidad generado automáticamente por el script (una línea JSON por consulta, con las fuentes usadas y la respuesta).
## Requisitos
 
- Python 3.9 o superior
- (Opcional, para respuestas reales del LLM) Librería `anthropic`:
```bash
  pip install anthropic
```
 
## Cómo ejecutar el sistema
 
1. Clona o descarga este repositorio.
2. (Opcional) Si quieres respuestas generadas por el LLM real en vez de una respuesta simulada, configura tu clave de API de Anthropic como variable de entorno:
```bash
   export ANTHROPIC_API_KEY="tu-api-key-aquí"
```
   Si no configuras esta variable, el script funciona igual pero entrega una **respuesta simulada**, lo que permite probar todo el flujo (recuperación de contexto, ensamblado de prompt, registro de logs) sin costo de API.
3. Ejecuta el script:
```bash
   python3 asistente_rag.py
```
4. El script correrá automáticamente 3 consultas de prueba (agendamiento, horario/ubicación, y una consulta clínica que debe ser derivada a un veterinario) e imprimirá en consola las fuentes recuperadas y la respuesta para cada una.
5. Cada consulta queda registrada en `logs_consultas.jsonl`, en la misma carpeta donde se ejecuta el script, para efectos de trazabilidad.
## Cómo probar con tus propias consultas
 
Importa la función `responder()` desde el script:
 
```python
from asistente_rag import responder
 
respuesta, contexto = responder("¿Atienden conejos?")
print(respuesta)
```
 
## Notas de diseño
 
- La "base vectorial" está simulada mediante una lista en memoria con recuperación por coincidencia de palabras clave (mock de embeddings), para efectos de evaluación. En un entorno de producción se reemplazaría por una base vectorial real (Chroma/FAISS) con embeddings, tal como se documenta en el informe técnico (sección 3).
- El asistente tiene reglas explícitas (guardrails en el `SYSTEM_PROMPT`) para **no** entregar diagnósticos, dosis ni interpretación de síntomas, derivando siempre esas consultas a un veterinario humano.
## Documento técnico
 
El informe completo con el análisis del caso, diseño del pipeline RAG, arquitectura y justificación de decisiones se entrega por separado en formato Word/PDF, según lo solicitado en la pauta de evaluación.
 
## Integrantes : 
- María Calfileo
- Stephanie Chamorro
- Leandro Marin

## Docente: 
- Juan Burgueño
