"""Configuración del LLM (OpenRouter vía API compatible con OpenAI).

La API key se lee SOLO de la variable de entorno OPENROUTER_API_KEY.
Nunca se escribe en el código ni en el repositorio.
"""
import os

MODELO = os.environ.get("OPENROUTER_MODEL", "openai/gpt-6-luna")
BASE_URL = "https://openrouter.ai/api/v1"

# Estrategia de ventana de contexto: se envían al LLM solo los últimos
# MAX_TOKENS_CONTEXTO tokens (aprox.) del historial; el historial completo
# sigue guardado en SQLite (checkpointer).
MAX_TOKENS_CONTEXTO = int(os.environ.get("MAX_TOKENS_CONTEXTO", "1500"))
LIMITE_PASOS = 14  # recursion_limit del grafo: evita bucles de tools


def crear_llm():
    from langchain_openai import ChatOpenAI

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "Falta OPENROUTER_API_KEY. Ejecuta: export OPENROUTER_API_KEY='tu-key'"
        )
    return ChatOpenAI(
        model=MODELO,
        base_url=BASE_URL,
        api_key=key,
        temperature=0,
        max_tokens=600,
        # esfuerzo de razonamiento bajo para cuidar el presupuesto de 1 USD
        extra_body={"reasoning": {"effort": "low"}},
    )
