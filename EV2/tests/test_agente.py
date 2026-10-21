
"""Pruebas de la orquestación del agente (ejecutar: python -m pytest -v tests)."""
import json
import os
import sys
from datetime import date
 
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
 
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
 
from agente_patitas.agente import AgentePatitas  # noqa: E402
from agente_patitas.tools import clasificar_urgencia  # noqa: E402
from agente_patitas.grafo import parsear_plan, ventana_de_contexto  # noqa: E402
from tests.llm_falso import LLMGuionado, llamar, texto  # noqa: E402
 
HOY = date(2026, 10, 20)  # martes -> día 0 de agenda = 2026-10-21
 
 
def plan(*pasos):
    return texto(json.dumps([{"paso": p, "prioridad": pr, "herramienta": h} for p, pr, h in pasos]))
 
 
def nombres_tools(traza):
    return [t for t in traza if t.startswith("TOOL CALL")]
 
 
def test_consulta_usa_rag(tmp_path):
    llm = LLMGuionado(respuestas=[
        plan(("buscar horario", 1, "buscar_conocimiento")),
        llamar("buscar_conocimiento", consulta="horario de atención"),
        texto("Atendemos de lunes a viernes de 09:00 a 19:00."),
    ])
    ag, traza = AgentePatitas(llm, str(tmp_path), hoy=HOY), []
    r = ag.conversar("¿A qué hora abren?", traza=traza)
    assert "09:00" in r
    assert any("horario_ubicacion" in t for t in traza if t.startswith("TOOL RESULT"))
 
 
def test_replanifica_si_hora_ocupada(tmp_path):
    ag0 = AgentePatitas(LLMGuionado(respuestas=[]), str(tmp_path), hoy=HOY)
    dia = ag0.agenda.dias[0]  # 10:00 ocupada por datos semilla
    ag0.cerrar()
 
    def tomar_alternativa(msgs):
        err = json.loads([m for m in msgs if isinstance(m, ToolMessage)][-1].content)
        assert err["ok"] is False
        return llamar("agendar_cita", "c2", mascota="Luna", especie="gato", servicio="control",
                      fecha=dia, hora=err["alternativas_mismo_dia"][0])
 
    llm = LLMGuionado(respuestas=[
        plan(("agendar control", 1, "agendar_cita")),
        llamar("agendar_cita", mascota="Luna", especie="gato", servicio="control", fecha=dia, hora="10:00"),
        tomar_alternativa,
        texto("Las 10:00 estaban ocupadas; agendé la primera hora libre."),
    ])
    ag = AgentePatitas(llm, str(tmp_path), hoy=HOY)
    ag.conversar("Quiero hora para Luna mañana a las 10")
    citas = [c for c in ag.agenda.citas.values() if c["mascota"] == "Luna"]
    assert len(citas) == 1 and citas[0]["hora"] != "10:00" and citas[0]["estado"] == "confirmada"
 
 
def test_urgencia_deriva_por_triaje_aunque_el_llm_no_llame_tools(tmp_path):
    """El triaje es código determinista: crea el ticket aunque el LLM no use ninguna tool."""
    def verifica_prompt(msgs):
        assert "YA registró la derivación" in msgs[0].content  # el agente sabe que ya se derivó
        return texto("Es urgente: lleva a Toby ahora a una clínica veterinaria.")
 
    llm = LLMGuionado(respuestas=[
        plan(("evaluar urgencia", 1, "evaluar_urgencia"), ("derivar", 2, "derivar_a_veterinario")),
        verifica_prompt,
    ])
    ag, traza = AgentePatitas(llm, str(tmp_path), hoy=HOY), []
    ag.conversar("Mi perro comió chocolate y vomita", traza=traza)
    assert any(t.startswith("TRIAJE (reglas): URGENTE") for t in traza)
    assert any(t.startswith("PLAN: [P1] evaluar urgencia") for t in traza)  # plan ordenado por prioridad
    tickets = [json.loads(l) for l in open(tmp_path / "derivaciones.jsonl")]
    assert len(tickets) == 1 and tickets[0]["nivel"] == "URGENTE"
    assert not any(c["mascota"] != "Reservado" for c in ag.agenda.citas.values())
 
 
def test_rutina_no_crea_ticket(tmp_path):
    llm = LLMGuionado(respuestas=[plan(("responder", 1, "buscar_conocimiento")), texto("Hola")])
    ag = AgentePatitas(llm, str(tmp_path), hoy=HOY)
    ag.conversar("¿Atienden conejos?")
    assert not os.path.exists(tmp_path / "derivaciones.jsonl")
 
 
def test_memoria_largo_plazo_entre_sesiones(tmp_path):
    llm1 = LLMGuionado(respuestas=[
        plan(("guardar dato", 1, "guardar_dato_memoria")),
        llamar("guardar_dato_memoria", dato="La dueña se llama Ana y su gata Luna es alérgica a la penicilina"),
        texto("Anotado, Ana."),
    ])
    a1 = AgentePatitas(llm1, str(tmp_path), hoy=HOY)
    a1.conversar("Soy Ana, mi gata Luna es alérgica a la penicilina", thread_id="sesion-A")
    a1.cerrar()
 
    def verifica_recuerdo(msgs):  # el system prompt de la nueva sesión debe traer el recuerdo
        assert "Luna" in msgs[0].content and "penicilina" in msgs[0].content
        return texto("Hola Ana, ¿es para Luna?")
 
    llm2 = LLMGuionado(respuestas=[plan(("saludar", 1, "buscar_conocimiento")), verifica_recuerdo])
    a2 = AgentePatitas(llm2, str(tmp_path), hoy=HOY)  # "reinicio": instancia nueva, thread nuevo
    assert "Luna" in a2.conversar("Quiero pedir una hora para mi gata", thread_id="sesion-B")
 
 
def test_historial_persiste_en_sqlite(tmp_path):
    llm = LLMGuionado(respuestas=[plan(("responder", 1, "buscar_conocimiento")), texto("Hola")])
    a = AgentePatitas(llm, str(tmp_path), hoy=HOY)
    a.conversar("hola", thread_id="t1")
    a.cerrar()
    b = AgentePatitas(LLMGuionado(respuestas=[]), str(tmp_path), hoy=HOY)
    msgs = b.grafo.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    assert [m.type for m in msgs] == ["human", "ai"]
 
 
def test_ventana_de_contexto_recorta_y_empieza_en_humano():
    historial = []
    for i in range(40):
        historial += [HumanMessage(content=f"pregunta {i} " + "x" * 200), AIMessage(content=f"resp {i} " + "y" * 200)]
    v = ventana_de_contexto(historial, max_tokens=500)
    assert len(v) < len(historial) and isinstance(v[0], HumanMessage)
    assert v[-1].content == historial[-1].content  # conserva lo más reciente
 
 
def test_ventana_conserva_turno_actual_aunque_exceda_presupuesto():
    historial = [HumanMessage(content="a" * 50), AIMessage(content="b" * 50), HumanMessage(content="z" * 4000)]
    v = ventana_de_contexto(historial, max_tokens=100)
    assert v and isinstance(v[0], HumanMessage) and v[-1].content.startswith("z")
 
 
def test_parsear_plan_tolerante():
    assert parsear_plan("texto basura") == []
    p = parsear_plan('Claro: [{"paso":"b","prioridad":2},{"paso":"a","prioridad":1}] listo')
    assert [x["paso"] for x in p] == ["a", "b"]
 
 
def test_clasificar_urgencia_sin_falsos_positivos():
    assert clasificar_urgencia("¿Atienden gatos y hasta qué hora están abiertos?") == "RUTINA"
    assert clasificar_urgencia("Mi perro tiene tos") == "MODERADA"
    assert clasificar_urgencia("Está vomitando") == "MODERADA"
    assert clasificar_urgencia("Se comió chocolate") == "URGENTE"
    assert clasificar_urgencia("Lo atropellaron") == "URGENTE"
 