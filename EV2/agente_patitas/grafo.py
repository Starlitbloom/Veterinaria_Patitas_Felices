"""Grafo del agente (LangGraph). Flujo por cada mensaje del usuario:

    recordar -> triaje -> planificar -> agente <-> herramientas -> FIN

- recordar:    recupera memoria de largo plazo relevante (semántica, TF-IDF).
- triaje:      reglas fijas (sin LLM): si hay señales urgentes crea el ticket de derivación.
- planificar:  el LLM descompone la petición en pasos con prioridad; el código
               ordena los pasos por prioridad (IE5).
- agente:      LLM con tools; ve el plan, los recuerdos y una ventana recortada
               del historial. Decide qué tool llamar (tool calling nativo).
- herramientas: ToolNode ejecuta las tools; el resultado vuelve al agente, que
               puede replanificar si un paso falla (ej. hora ocupada).
"""
import json
import re
from datetime import date
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .config import MAX_TOKENS_CONTEXTO
from .tools import clasificar_urgencia

SYSTEM_PROMPT = """Eres el agente virtual de la Veterinaria Patitas Felices (Rancagua).
Hoy es {hoy}. Atiendes de lunes a viernes 09:00-19:00.

REGLAS
- Información de la clínica: SOLO la que entrega buscar_conocimiento. No inventes.
- NUNCA entregues diagnósticos, dosis ni interpretes síntomas. Ante síntomas o accidentes:
  usa evaluar_urgencia; si es URGENTE deriva de inmediato con derivar_a_veterinario
  (no agendes rutina); si es MODERADA deriva y ofrece la primera hora libre.
- Para agendar: consulta disponibilidad, agenda con agendar_cita y confirma con el id.
  Si una hora está ocupada, NO te rindas: propón o toma las alternativas que devuelve la tool.
- Cuando el dueño cuente datos durables (nombre, mascotas, alergias, preferencias),
  guárdalos con guardar_dato_memoria.
- Los recuerdos del dueño son hechos confirmados. Si alguno es relevante (mascotas, alergias,
  preferencias), úsalo y menciónalo explícitamente al responder, sobre todo al agendar o
  cuando el dueño pregunte si debe considerar algo.
- Nunca afirmes que hiciste una acción (agendar, cancelar, derivar, guardar) si no llamaste a la
  tool correspondiente en este turno, salvo que el triaje automático indique que ya se registró.
- Responde en español, cercano y breve (máx. 5 líneas).

TRIAJE AUTOMÁTICO (lo ejecuta el sistema con reglas fijas antes de tu turno):
{triaje}

RECUERDOS DEL DUEÑO (memoria de largo plazo):
{memorias}

PLAN DE LA TAREA (sigue el orden, ajústalo si algo falla):
{plan}
"""

PROMPT_PLAN = """Descompón la petición del dueño de mascota en pasos ordenados.
Responde SOLO un JSON: [{{"paso": "...", "prioridad": 1-3, "herramienta": "..."}}]
prioridad 1 = más urgente. Si hay síntomas o accidente, el primer paso es evaluar_urgencia
con prioridad 1 y derivar_a_veterinario si corresponde. Herramientas posibles:
{herramientas}.
Petición: {peticion}"""


class Estado(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list
    memorias: list
    triaje: dict


TEXTO_TRIAJE = {
    "URGENTE": "Nivel URGENTE. El sistema YA registró la derivación a un veterinario. No agendes citas "
               "de rutina. Indica que lleve o llame a la clínica de inmediato; no des dosis ni diagnóstico.",
    "MODERADA": "Nivel MODERADA. Deriva con derivar_a_veterinario y ofrece la primera hora libre.",
    "RUTINA": "Sin señales de urgencia.",
}


def _tokens_aprox(mensajes) -> int:
    """Contador aproximado (4 caracteres ~ 1 token), sin dependencias extra."""
    return sum(len(str(getattr(m, "content", ""))) // 4 + 4 for m in mensajes)


def ventana_de_contexto(mensajes: list, max_tokens: int = MAX_TOKENS_CONTEXTO) -> list:
    """Estrategia de ventana deslizante: conserva los mensajes más recientes
    que caben en max_tokens, empezando siempre en un mensaje humano para no
    cortar pares tool_call/tool_result. Si el turno actual por sí solo excede
    el presupuesto, se conserva completo (nunca se pierde el turno en curso)."""
    recortado = trim_messages(mensajes, max_tokens=max_tokens, token_counter=_tokens_aprox,
                              strategy="last", start_on="human", allow_partial=False)
    if recortado:
        return recortado
    for i in range(len(mensajes) - 1, -1, -1):
        if isinstance(mensajes[i], HumanMessage):
            return mensajes[i:]
    return mensajes


def parsear_plan(texto: str) -> list:
    """Extrae el JSON del plan y lo ordena por prioridad; tolera respuestas sucias."""
    try:
        m = re.search(r"\[.*\]", texto, re.S)
        pasos = json.loads(m.group(0)) if m else []
        pasos = [p for p in pasos if isinstance(p, dict) and "paso" in p]
        return sorted(pasos, key=lambda p: p.get("prioridad", 3))
    except (json.JSONDecodeError, TypeError):
        return []


def construir_grafo(llm, tools: list, memoria, dueno_id: str, checkpointer=None, hoy: date = None):
    hoy = hoy or date.today()
    llm_con_tools = llm.bind_tools(tools)
    nombres = ", ".join(t.name for t in tools)

    def recordar(estado: Estado):
        texto = estado["messages"][-1].content
        return {"memorias": memoria.recuperar(dueno_id, str(texto))}

    por_nombre = {t.name: t for t in tools}

    def triaje(estado: Estado):
        """Decisión de seguridad determinista (sin LLM): si el mensaje tiene señales
        urgentes, se crea el ticket de derivación en código, así el agente no puede
        olvidarlo ni afirmar que derivó sin haberlo hecho."""
        texto = str(estado["messages"][-1].content)
        nivel = clasificar_urgencia(texto)
        if nivel == "URGENTE":
            por_nombre["derivar_a_veterinario"].invoke({"motivo": texto[:300], "nivel": "URGENTE"})
        return {"triaje": {"nivel": nivel, "ticket": nivel == "URGENTE"}}

    def planificar(estado: Estado):
        peticion = str(estado["messages"][-1].content)
        resp = llm.invoke([SystemMessage(content=PROMPT_PLAN.format(herramientas=nombres, peticion=peticion))])
        return {"plan": parsear_plan(str(resp.content))}

    def agente(estado: Estado):
        plan_txt = "\n".join(f"{i}. [P{p.get('prioridad', 3)}] {p['paso']}"
                             for i, p in enumerate(estado.get("plan") or [], 1)) or "(sin plan)"
        mem_txt = "\n".join(f"- {m}" for m in estado.get("memorias") or []) or "(sin recuerdos)"
        sistema = SystemMessage(content=SYSTEM_PROMPT.format(hoy=hoy.isoformat(), memorias=mem_txt, plan=plan_txt,
                                                          triaje=TEXTO_TRIAJE[(estado.get("triaje") or {}).get("nivel", "RUTINA")]))
        ventana = ventana_de_contexto(estado["messages"])
        return {"messages": [llm_con_tools.invoke([sistema] + ventana)]}

    g = StateGraph(Estado)
    g.add_node("recordar", recordar)
    g.add_node("triaje", triaje)
    g.add_node("planificar", planificar)
    g.add_node("agente", agente)
    g.add_node("herramientas", ToolNode(tools))
    g.add_edge(START, "recordar")
    g.add_edge("recordar", "triaje")
    g.add_edge("triaje", "planificar")
    g.add_edge("planificar", "agente")
    g.add_conditional_edges("agente", tools_condition, {"tools": "herramientas", END: END})
    g.add_edge("herramientas", "agente")
    return g.compile(checkpointer=checkpointer)
