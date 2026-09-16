"""
Asistente Virtual RAG - Veterinaria Patitas Felices
------------------------------------------------
Prototipo funcional que simula el pipeline RAG descrito en el informe:
1) Ingesta y troceo de documentos internos (FAQ, precios, protocolos).
2) Indexación simple por similitud de palabras clave (mock de embeddings).
3) Recuperación de los fragmentos más relevantes según la consulta.
4) Ensamblado del prompt (system + contexto recuperado + pregunta).
5) Generación de la respuesta vía LLM (API de Claude / Anthropic).
6) Registro de la consulta en un log para trazabilidad.

Nota: para efectos de evaluación, la "base vectorial" se simula con una
lista en memoria. En producción se reemplazaría por Chroma/FAISS + embeddings reales,
tal como se documenta en el informe técnico (sección 3, IE3).
"""

import json
import os
import re
from datetime import datetime


STOPWORDS = {
    "el", "la", "los", "las", "de", "del", "en", "y", "a", "que", "qué", "es",
    "un", "una", "para", "por", "con", "mi", "mis", "su", "sus", "me", "se",
    "puedo", "puede", "cómo", "como", "dónde", "donde", "cuál", "cual", "más",
    "si", "no", "le", "lo", "al", "o", "sobre", "hay",
}


def _tokenizar(texto: str):
    """Normaliza, separa en palabras y quita puntuación + stopwords básicas."""
    palabras = re.findall(r"[a-záéíóúñ0-9]+", texto.lower())
    return {p for p in palabras if p not in STOPWORDS and len(p) > 2}

# ---------------------------------------------------------------------------
# 1) FUENTES INTERNAS (simulan los documentos reales de la clínica)
# ---------------------------------------------------------------------------
FUENTES_INTERNAS = [
    {
        "id": "faq_agendar_cita",
        "texto": "¿Cómo agendo una hora para mi mascota? Puedes solicitar tu cita en línea "
                 "desde el botón 'Solicitar cita', eligiendo el servicio, la fecha y la "
                 "hora que más te acomoden. Nuestro equipo confirma la solicitud a la brevedad.",
    },
    {
        "id": "faq_especies",
        "texto": "¿Qué especies atienden además de perros y gatos? Además de perros y gatos, "
                 "atendemos conejos y aves. Puedes ver el detalle de servicios disponibles "
                 "por especie en la sección de servicios.",
    },
    {
        "id": "faq_historial_clinico",
        "texto": "¿Puedo ver el historial clínico de mi mascota en línea? Sí. Una vez que "
                 "creas tu cuenta, puedes revisar el historial de consultas, vacunas y "
                 "tratamientos de tu mascota desde tu perfil, sin necesidad de llamar a la clínica.",
    },
    {
        "id": "faq_reprogramar_cita",
        "texto": "¿Qué pasa si no puedo asistir a mi hora? Puedes reagendar o cancelar tu "
                 "cita desde 'Mis citas' con anticipación, para que ese horario quede "
                 "disponible para otra mascota que lo necesite.",
    },
    {
        "id": "horario_ubicacion",
        "texto": "Veterinaria Patitas Felices atiende de lunes a viernes de 09:00 a 19:00 hrs. "
                 "Está ubicada en Rancagua, Región de O'Higgins.",
    },
    {
        "id": "catalogo_vacunas",
        "texto": "Catálogo de vacunas de la clínica: Vacuna antirrábica canina (VA001), "
                 "Vacuna séxtuple canina (VA002), Vacuna bivalente felina (VA003), "
                 "Vacuna triple felina (VA004), Vacuna Bordetella canina (VA005), "
                 "Vacuna antirrábica felina (VA006).",
    },
]

# Fuente externa simulada (normativa / guía pública)
FUENTES_EXTERNAS = [
    {
        "id": "guia_sag_vacunacion",
        "texto": "El Servicio Agrícola y Ganadero (SAG) recomienda mantener al día "
                 "el esquema de vacunación antirrábica en caninos y felinos como medida "
                 "de salud pública.",
    },
]

SYSTEM_PROMPT = (
    "Eres el asistente virtual de la Veterinaria Patitas Felices (Rancagua). "
    "Respondes únicamente sobre horarios, servicios, precios, protocolos de "
    "vacunación y proceso de agendamiento de la clínica, usando solo la "
    "información recuperada del contexto (RAG). Nunca entregues diagnósticos, "
    "indicaciones de dosis de medicamentos ni interpretación de síntomas: ante "
    "cualquier consulta clínica, indica que debe ser evaluada por un veterinario "
    "y ofrece agendar hora. Responde en español, en tono cercano y breve "
    "(máx. 4 líneas), citando la fuente interna cuando corresponda."
)

LOG_PATH = "logs_consultas.jsonl"


def recuperar_contexto(consulta: str, top_k: int = 2):
    """Retriever simplificado: recupera los fragmentos cuyo texto comparte
    más palabras clave con la consulta. Simula la búsqueda semántica descrita
    en el informe (etapa 3: Recuperación)."""
    todas_las_fuentes = FUENTES_INTERNAS + FUENTES_EXTERNAS
    palabras_consulta = _tokenizar(consulta)

    puntuadas = []
    for doc in todas_las_fuentes:
        palabras_doc = _tokenizar(doc["texto"])
        score = len(palabras_consulta & palabras_doc)
        puntuadas.append((score, doc))

    puntuadas.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in puntuadas[:top_k] if score > 0]


def ensamblar_prompt(consulta: str, contexto: list):
    """Ensambla el prompt final: system + contexto recuperado + pregunta."""
    contexto_texto = "\n".join(f"- [{d['id']}] {d['texto']}" for d in contexto)
    prompt_usuario = (
        f"Contexto recuperado:\n{contexto_texto}\n\n"
        f"Pregunta del usuario: {consulta}"
    )
    return SYSTEM_PROMPT, prompt_usuario


def generar_respuesta(system_prompt: str, prompt_usuario: str):
    """Llama al LLM (API de Claude) para generar la respuesta final.
    Requiere la variable de entorno ANTHROPIC_API_KEY.
    Si no está configurada, retorna una respuesta simulada para poder
    probar el flujo completo sin costo de API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        return (
            "[RESPUESTA SIMULADA - configura ANTHROPIC_API_KEY para respuesta real]\n"
            "Basado en el contexto recuperado, aquí iría la respuesta generada por el LLM."
        )

    import anthropic  # pip install anthropic

    client = anthropic.Anthropic(api_key=api_key)
    mensaje = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt_usuario}],
    )
    return mensaje.content[0].text


def registrar_log(consulta: str, contexto: list, respuesta: str):
    """Guarda la consulta, el contexto usado y la respuesta para trazabilidad
    (ver sección 3, etapa 6 del informe: Registro)."""
    entrada = {
        "timestamp": datetime.now().isoformat(),
        "consulta": consulta,
        "fuentes_usadas": [d["id"] for d in contexto],
        "respuesta": respuesta,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entrada, ensure_ascii=False) + "\n")


def responder(consulta: str):
    contexto = recuperar_contexto(consulta)
    system_prompt, prompt_usuario = ensamblar_prompt(consulta, contexto)
    respuesta = generar_respuesta(system_prompt, prompt_usuario)
    registrar_log(consulta, contexto, respuesta)
    return respuesta, contexto


if __name__ == "__main__":
    consultas_prueba = [
        "¿Cómo agendo una hora para mi mascota?",
        "¿Cuál es el horario de atención y dónde está ubicada la clínica?",
        "Mi gato tiene diarrea, ¿qué le puedo dar?",  # debe derivar a veterinario
    ]

    for c in consultas_prueba:
        print(f"\nConsulta: {c}")
        respuesta, contexto = responder(c)
        print(f"Fuentes usadas: {[d['id'] for d in contexto]}")
        print(f"Respuesta: {respuesta}")
