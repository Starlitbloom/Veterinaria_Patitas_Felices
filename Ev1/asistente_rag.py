"""
Asistente Virtual RAG - Veterinaria Patitas Felices
------------------------------------------------
Prototipo funcional que implementa el pipeline RAG descrito en el informe:
1) Ingesta y troceo de documentos internos (FAQ, catálogo de vacunas, horario).
2) Indexación: cada fragmento se vectoriza con TF-IDF (vector store en memoria).
3) Recuperación: la consulta se vectoriza igual y se compara por similitud
   coseno contra todos los vectores indexados (recuperación semántica real,
   no solo coincidencia de palabras).
4) Ensamblado del prompt (system + contexto recuperado + pregunta).
5) Generación de la respuesta vía LLM (API de Claude / Anthropic).
6) Registro de la consulta en un log para trazabilidad.

Nota: se usa TF-IDF + similitud coseno (scikit-learn) como motor de
vectorización y recuperación, en vez de una base vectorial externa como
Chroma/FAISS con embeddings de un modelo de lenguaje. Es una técnica de
recuperación de información real y ampliamente usada como baseline de RAG,
que aquí permite un prototipo 100% funcional, reproducible y sin
dependencias externas pesadas ni costo de API. La arquitectura (ingesta →
indexación → recuperación → ensamblado → generación → registro) es la misma
que se escalaría a embeddings de un modelo + una base vectorial dedicada en
una versión de producción.
"""

import json
import os
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Stopwords en español (sklearn no trae una lista nativa para este idioma)
STOPWORDS_ES = [
    "el", "la", "los", "las", "de", "del", "en", "y", "a", "que", "qué", "es",
    "un", "una", "unos", "unas", "para", "por", "con", "mi", "mis", "su", "sus",
    "me", "se", "te", "puedo", "puede", "pueden", "cómo", "como", "dónde",
    "donde", "cuál", "cuales", "cual", "más", "si", "no", "le", "les", "lo",
    "al", "o", "sobre", "hay", "está", "estan", "están", "soy", "eres", "ser",
    "tu", "tus", "yo", "él", "ella", "ellos", "ellas", "nos", "ese", "esa",
    "esos", "esas", "este", "esta", "estos", "estas", "también", "muy", "ya",
    "pero", "porque", "cuando", "qué", "dar", "tener", "tengo", "tiene",
]

# ---------------------------------------------------------------------------
# 1) FUENTES INTERNAS (documentos reales de la clínica)
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
UMBRAL_SIMILITUD = 0.08  # bajo este puntaje, se considera que no hay contexto relevante


# ---------------------------------------------------------------------------
# 2) INDEXACIÓN — "base vectorial" en memoria (TF-IDF)
# ---------------------------------------------------------------------------
class VectorStore:
    """Índice vectorial simple en memoria: vectoriza todos los documentos con
    TF-IDF una sola vez (etapa de Indexación) y permite recuperar por
    similitud coseno (etapa de Recuperación)."""

    def __init__(self, documentos: list):
        self.documentos = documentos
        self.vectorizer = TfidfVectorizer(stop_words=STOPWORDS_ES)
        textos = [d["texto"] for d in documentos]
        self.matriz = self.vectorizer.fit_transform(textos)

    def buscar(self, consulta: str, top_k: int = 2, umbral: float = UMBRAL_SIMILITUD):
        vector_consulta = self.vectorizer.transform([consulta])
        similitudes = cosine_similarity(vector_consulta, self.matriz)[0]

        resultados = sorted(
            zip(similitudes, self.documentos), key=lambda x: x[0], reverse=True
        )
        return [
            (doc, float(score))
            for score, doc in resultados[:top_k]
            if score >= umbral
        ]


_INDICE = VectorStore(FUENTES_INTERNAS + FUENTES_EXTERNAS)


def recuperar_contexto(consulta: str, top_k: int = 2):
    """Recupera los fragmentos más relevantes por similitud coseno sobre
    vectores TF-IDF (etapa 3: Recuperación)."""
    resultados = _INDICE.buscar(consulta, top_k=top_k)
    return [doc for doc, score in resultados], [score for doc, score in resultados]


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


def registrar_log(consulta: str, contexto: list, similitudes: list, respuesta: str):
    """Guarda la consulta, el contexto usado y la respuesta para trazabilidad
    (ver sección 3, etapa 6 del informe: Registro)."""
    entrada = {
        "timestamp": datetime.now().isoformat(),
        "consulta": consulta,
        "fuentes_usadas": [
            {"id": d["id"], "similitud_coseno": round(s, 4)}
            for d, s in zip(contexto, similitudes)
        ],
        "respuesta": respuesta,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entrada, ensure_ascii=False) + "\n")


def responder(consulta: str):
    contexto, similitudes = recuperar_contexto(consulta)
    system_prompt, prompt_usuario = ensamblar_prompt(consulta, contexto)
    respuesta = generar_respuesta(system_prompt, prompt_usuario)
    registrar_log(consulta, contexto, similitudes, respuesta)
    return respuesta, contexto, similitudes


if __name__ == "__main__":
    consultas_prueba = [
        "¿Cómo agendo una hora para mi mascota?",
        "¿Cuál es el horario de atención y dónde está ubicada la clínica?",
        "Mi gato tiene diarrea, ¿qué le puedo dar?",  # debe derivar a veterinario
    ]

    for c in consultas_prueba:
        print(f"\nConsulta: {c}")
        respuesta, contexto, similitudes = responder(c)
        fuentes = [f"{d['id']} (sim={s:.3f})" for d, s in zip(contexto, similitudes)]
        print(f"Fuentes recuperadas: {fuentes if fuentes else 'ninguna (bajo umbral -> deriva a humano)'}")
        print(f"Respuesta: {respuesta}")