"""Punto de entrada del agente.

  python main.py            # chat interactivo
  python main.py --demo     # 4 escenarios de la rúbrica (autonomía, replanificación, urgencia, memoria)

Requiere: export OPENROUTER_API_KEY='...'
"""
import argparse
import sys

from agente_patitas.agente import AgentePatitas
from agente_patitas.config import MODELO, crear_llm


def escenarios(ag):
    dia = ag.agenda.dias[0]
    return [
        ("1. Consulta informativa (tool de consulta/RAG)", "sesion-1",
         "¿Atienden conejos y hasta qué hora están abiertos?"),
        ("2. Agendamiento con conflicto -> replanificación (tools de escritura)", "sesion-1",
         f"Quiero una hora de control para mi gata Luna el {dia} a las 10:00."),
        ("3. Urgencia -> el plan cambia de prioridad y deriva", "sesion-2",
         "Mi perro Toby se comió una barra de chocolate y está vomitando, ¿qué le doy?"),
        ("4a. Memoria de largo plazo: dato durable", "sesion-3",
         "Hola, soy Ana. Mi gata Luna es alérgica a la penicilina, tenlo presente."),
        ("4b. Memoria de largo plazo: NUEVA sesión (otro thread_id)", "sesion-4",
         "Quiero pedir hora para mi gata, ¿tienes algo que deba considerar?"),
    ]


def correr_demo(ag):
    for titulo, thread, msg in escenarios(ag):
        print(f"\n{'=' * 78}\n{titulo}\n{'=' * 78}\nUSUARIO ({thread}): {msg}")
        traza = []
        r = ag.conversar(msg, thread_id=thread, traza=traza)
        for t in traza:
            print("  ·", t)
        print(f"AGENTE: {r}")


def chat(ag):
    print(f"Agente Patitas Felices ({MODELO}). Escribe 'salir' para terminar.")
    thread = input("ID de sesión [sesion-1]: ").strip() or "sesion-1"
    while True:
        msg = input("\nTú: ").strip()
        if msg.lower() in ("salir", "exit", ""):
            break
        traza = []
        r = ag.conversar(msg, thread_id=thread, traza=traza)
        for t in traza:
            print("  ·", t)
        print(f"Agente: {r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--datos", default="datos")
    a = p.parse_args()
    try:
        agente = AgentePatitas(crear_llm(), datos_dir=a.datos)
    except RuntimeError as e:
        sys.exit(str(e))
    try:
        correr_demo(agente) if a.demo else chat(agente)
    finally:
        agente.cerrar()
