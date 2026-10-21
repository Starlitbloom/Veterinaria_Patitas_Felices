"""Agenda de la clínica (sistema simulado de agendamiento, persistente en JSON).

Es la "herramienta de escritura" principal: el agente modifica este estado.
"""
import json
import os
from datetime import date, timedelta

HORAS = [f"{h:02d}:00" for h in range(9, 19)]  # 09:00 .. 18:00 (cierra 19:00)
ESPECIES = {"perro", "gato", "conejo", "ave"}


def proximos_dias_habiles(hoy: date, n: int = 5) -> list:
    dias, d = [], hoy
    while len(dias) < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            dias.append(d.isoformat())
    return dias


class Agenda:
    def __init__(self, ruta: str, hoy: date = None):
        self.ruta = ruta
        self.hoy = hoy or date.today()
        self.dias = proximos_dias_habiles(self.hoy)
        if os.path.exists(ruta):
            with open(ruta, encoding="utf-8") as f:
                self.citas = json.load(f)
        else:
            self.citas = {}
            # Datos semilla: horas ocupadas para poder demostrar replanificación
            for h in ("10:00", "11:00", "15:00"):
                self._ocupar(self.dias[0], h, "Reservado", "perro", "control")
            for h in ("09:00", "10:00", "11:00", "12:00"):
                self._ocupar(self.dias[1], h, "Reservado", "gato", "control")
            self._guardar()

    def _ocupar(self, fecha, hora, mascota, especie, servicio):
        id_cita = f"C{len(self.citas) + 1:03d}"
        self.citas[id_cita] = {"id": id_cita, "fecha": fecha, "hora": hora,
                               "mascota": mascota, "especie": especie,
                               "servicio": servicio, "estado": "confirmada"}
        return self.citas[id_cita]

    def _guardar(self):
        os.makedirs(os.path.dirname(self.ruta) or ".", exist_ok=True)
        with open(self.ruta, "w", encoding="utf-8") as f:
            json.dump(self.citas, f, ensure_ascii=False, indent=2)

    def horas_libres(self, fecha: str) -> list:
        ocupadas = {c["hora"] for c in self.citas.values()
                    if c["fecha"] == fecha and c["estado"] == "confirmada"}
        return [h for h in HORAS if h not in ocupadas]

    def agendar(self, mascota, especie, servicio, fecha, hora) -> dict:
        if especie.lower() not in ESPECIES:
            return {"ok": False, "error": f"Especie '{especie}' no atendida. Atendemos: {sorted(ESPECIES)}"}
        if fecha not in self.dias:
            return {"ok": False, "error": f"Fecha fuera de agenda. Fechas disponibles: {self.dias}"}
        libres = self.horas_libres(fecha)
        if hora not in libres:
            return {"ok": False, "error": f"Hora {hora} no disponible el {fecha}.",
                    "alternativas_mismo_dia": libres[:4],
                    "otros_dias": [d for d in self.dias if d != fecha and self.horas_libres(d)][:2]}
        cita = self._ocupar(fecha, hora, mascota, especie.lower(), servicio)
        self._guardar()
        return {"ok": True, "cita": cita}

    def cancelar(self, id_cita: str) -> dict:
        c = self.citas.get(id_cita)
        if not c or c["estado"] != "confirmada":
            return {"ok": False, "error": f"No existe una cita confirmada con id {id_cita}"}
        c["estado"] = "cancelada"
        self._guardar()
        return {"ok": True, "cita": c}
