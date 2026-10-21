"""Herramientas del agente, agrupadas por tipo (IE1).

CONSULTA:      buscar_conocimiento, consultar_disponibilidad, consultar_historial_mascota
RAZONAMIENTO:  evaluar_urgencia
ESCRITURA:     agendar_cita, cancelar_cita, guardar_dato_memoria, derivar_a_veterinario

Para agregar una tool nueva: definir una función con @tool (el docstring es
lo que el LLM lee para decidir cuándo usarla) y añadirla a la lista que
retorna crear_tools(). No hay que tocar el grafo.
"""
import json
import os
import re
from datetime import datetime

from langchain_core.tools import tool

from .conocimiento import INDICE_CLINICA

# Señales de urgencia (reglas explícitas: el agente NO diagnostica)
SENALES_URGENTES = [
    "chocolate", "veneno", "intoxic", "atropell", "convuls", "sangre", "sangra",
    "no respira", "dificultad para respirar", "desmay", "inconsciente", "fractura",
    "no puede orinar", "hinchazón abdominal", "golpe",
]
SENALES_MODERADAS = [
    "vomit", "diarrea", "cojea", "no come", "decaído", "fiebre", "letargo",
    "tos", "secreción", "rasca", "herida",
]

# Historial clínico simulado (equivale al "perfil" en la web de la clínica)
HISTORIAL = {
    "luna": {"especie": "gato", "vacunas": ["Triple felina (VA004) - 2026-03-12"],
             "ultima_consulta": "2026-03-12 control anual"},
    "toby": {"especie": "perro", "vacunas": ["Séxtuple canina (VA002) - 2025-11-02",
                                             "Antirrábica canina (VA001) - 2025-11-02"],
             "ultima_consulta": "2025-11-02 vacunación"},
}


def _coincide(senales: list, texto: str) -> bool:
    """Coincidencia por palabra (no por subcadena): 'tos' no debe activarse con
    'gatos' ni 'abiertos'. Las señales cortas exigen palabra completa; el resto
    se compara como inicio de palabra para admitir variantes ('vomit' -> vomita, vomitando)."""
    for senal in senales:
        patron = rf"\b{re.escape(senal)}\b" if len(senal) <= 4 else rf"\b{re.escape(senal)}"
        if re.search(patron, texto):
            return True
    return False


def clasificar_urgencia(descripcion: str) -> str:
    """Reglas fijas (no LLM): devuelve URGENTE, MODERADA o RUTINA."""
    t = descripcion.lower()
    if _coincide(SENALES_URGENTES, t):
        return "URGENTE"
    if _coincide(SENALES_MODERADAS, t):
        return "MODERADA"
    return "RUTINA"


def crear_tools(agenda, memoria, dueno_id: str, datos_dir: str) -> list:
    @tool
    def buscar_conocimiento(consulta: str) -> str:
        """Busca en la base de conocimiento de la clínica (horarios, ubicación,
        especies atendidas, vacunas, cómo agendar/reagendar, historial en línea).
        Úsala para cualquier pregunta informativa general."""
        res = INDICE_CLINICA.buscar(consulta, top_k=2)
        if not res:
            return "SIN_RESULTADOS: no hay información relevante. Ofrece derivar a un humano."
        return "\n".join(f"[{d['id']}] {d['texto']}" for d, _ in res)

    @tool
    def consultar_disponibilidad(fecha: str) -> str:
        """Lista las horas libres para una fecha (formato YYYY-MM-DD).
        Úsala ANTES de agendar o cuando una hora pedida esté ocupada."""
        if fecha not in agenda.dias:
            return json.dumps({"error": "fecha fuera de agenda", "fechas_validas": agenda.dias})
        return json.dumps({"fecha": fecha, "horas_libres": agenda.horas_libres(fecha)})

    @tool
    def consultar_historial_mascota(nombre_mascota: str) -> str:
        """Devuelve vacunas y última consulta registradas de una mascota del dueño."""
        h = HISTORIAL.get(nombre_mascota.strip().lower())
        return json.dumps(h, ensure_ascii=False) if h else "Mascota no encontrada en el historial."

    @tool
    def evaluar_urgencia(descripcion: str) -> str:
        """Clasifica la urgencia de lo que cuenta el dueño (URGENTE, MODERADA o RUTINA)
        con reglas fijas. NO es un diagnóstico. Úsala siempre que el dueño describa
        síntomas o un accidente de su mascota."""
        nivel = clasificar_urgencia(descripcion)
        return json.dumps({"nivel": nivel,
                           "accion": {"URGENTE": "derivar de inmediato a veterinario; no agendar rutina",
                                      "MODERADA": "derivar a veterinario y ofrecer primera hora disponible",
                                      "RUTINA": "continuar con flujo normal"}[nivel]})

    @tool
    def agendar_cita(mascota: str, especie: str, servicio: str, fecha: str, hora: str) -> str:
        """Agenda una cita (ESCRIBE en la agenda). fecha=YYYY-MM-DD, hora=HH:00.
        Si la hora está ocupada devuelve alternativas: replanifica con ellas."""
        return json.dumps(agenda.agendar(mascota, especie, servicio, fecha, hora), ensure_ascii=False)

    @tool
    def cancelar_cita(id_cita: str) -> str:
        """Cancela una cita confirmada por su id (ej. C004)."""
        return json.dumps(agenda.cancelar(id_cita), ensure_ascii=False)

    @tool
    def guardar_dato_memoria(dato: str) -> str:
        """Guarda en la memoria de largo plazo un dato durable del dueño o su mascota
        (nombre, mascotas, alergias, preferencias de horario). Redáctalo como frase
        completa. No guardes datos de un solo uso."""
        return json.dumps(memoria.guardar(dueno_id, dato), ensure_ascii=False)

    @tool
    def derivar_a_veterinario(motivo: str, nivel: str) -> str:
        """Crea un ticket de derivación a un veterinario humano (ESCRIBE en el registro).
        nivel: URGENTE, MODERADA o RUTINA."""
        os.makedirs(datos_dir, exist_ok=True)
        ticket = {"fecha": datetime.now().isoformat(timespec="seconds"), "dueno_id": dueno_id,
                  "nivel": nivel, "motivo": motivo}
        with open(os.path.join(datos_dir, "derivaciones.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(ticket, ensure_ascii=False) + "\n")
        return json.dumps({"ok": True, "ticket": ticket}, ensure_ascii=False)

    return [buscar_conocimiento, consultar_disponibilidad, consultar_historial_mascota,
            evaluar_urgencia, agendar_cita, cancelar_cita, guardar_dato_memoria,
            derivar_a_veterinario]
