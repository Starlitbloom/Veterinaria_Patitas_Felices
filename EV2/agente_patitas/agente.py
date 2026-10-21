"""Ensamblado del agente: memoria, agenda, tools, grafo y checkpointer SQLite."""
import os
import sqlite3
from datetime import date

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from .agenda import Agenda
from .config import LIMITE_PASOS
from .grafo import construir_grafo
from .memoria import MemoriaLargoPlazo
from .tools import crear_tools


class AgentePatitas:
    def __init__(self, llm, datos_dir: str = "datos", dueno_id: str = "dueno_demo", hoy: date = None):
        os.makedirs(datos_dir, exist_ok=True)
        self.datos_dir, self.dueno_id = datos_dir, dueno_id
        self.agenda = Agenda(os.path.join(datos_dir, "agenda.json"), hoy)
        self.memoria = MemoriaLargoPlazo(os.path.join(datos_dir, "memoria_largo_plazo.json"))
        self.tools = crear_tools(self.agenda, self.memoria, dueno_id, datos_dir)
        # Memoria de corto plazo: historial por thread_id persistido en SQLite
        self._conn = sqlite3.connect(os.path.join(datos_dir, "historial.sqlite"), check_same_thread=False)
        self.grafo = construir_grafo(llm, self.tools, self.memoria, dueno_id,
                                     checkpointer=SqliteSaver(self._conn), hoy=hoy)

    def conversar(self, texto: str, thread_id: str = "sesion-1", traza: list = None) -> str:
        """Envía un mensaje; si se pasa `traza`, se llena con los eventos del flujo."""
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": LIMITE_PASOS}
        ultimo = ""
        for evento in self.grafo.stream({"messages": [HumanMessage(content=texto)]}, config,
                                        stream_mode="updates"):
            for nodo, upd in evento.items():
                if traza is not None:
                    traza.append(self._resumir(nodo, upd))
                for m in (upd or {}).get("messages", []):
                    if m.type == "ai" and not getattr(m, "tool_calls", None):
                        ultimo = m.content
        return ultimo

    @staticmethod
    def _resumir(nodo, upd):
        upd = upd or {}
        if nodo == "planificar":
            return f"PLAN: " + " -> ".join(f"[P{p.get('prioridad')}] {p['paso']}" for p in upd.get("plan", []))
        if nodo == "triaje":
            t = upd.get("triaje", {})
            return f"TRIAJE (reglas): {t.get('nivel')}" + (" -> ticket de derivación creado" if t.get("ticket") else "")
        if nodo == "recordar":
            return f"MEMORIA RECUPERADA: {upd.get('memorias') or 'ninguna'}"
        for m in upd.get("messages", []):
            if m.type == "ai" and getattr(m, "tool_calls", None):
                return "TOOL CALL: " + "; ".join(f"{c['name']}({c['args']})" for c in m.tool_calls)
            if m.type == "tool":
                return f"TOOL RESULT [{m.name}]: {str(m.content)[:160]}"
        return f"{nodo}"

    def cerrar(self):
        self._conn.close()
