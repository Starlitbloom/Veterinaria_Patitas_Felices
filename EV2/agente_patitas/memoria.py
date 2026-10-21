"""Memoria de largo plazo con recuperación semántica (IE3, IE4).

- Se persiste en un archivo JSON (sobrevive entre sesiones/threads).
- Cada recuerdo pertenece a un dueño (dueno_id).
- La recuperación vectoriza los recuerdos del dueño con TF-IDF y devuelve
  los más similares a la consulta actual (similitud coseno).
"""
import json
import os
from datetime import datetime

from .conocimiento import IndiceTfidf


class MemoriaLargoPlazo:
    def __init__(self, ruta: str):
        self.ruta = ruta
        self._recuerdos = []
        if os.path.exists(ruta):
            with open(ruta, encoding="utf-8") as f:
                self._recuerdos = json.load(f)

    def guardar(self, dueno_id: str, texto: str) -> dict:
        if any(r["dueno_id"] == dueno_id and r["texto"] == texto for r in self._recuerdos):
            return {"estado": "ya_existia", "texto": texto}
        rec = {
            "id": f"m{len(self._recuerdos) + 1}",
            "dueno_id": dueno_id,
            "texto": texto,
            "fecha": datetime.now().isoformat(timespec="seconds"),
        }
        self._recuerdos.append(rec)
        os.makedirs(os.path.dirname(self.ruta) or ".", exist_ok=True)
        with open(self.ruta, "w", encoding="utf-8") as f:
            json.dump(self._recuerdos, f, ensure_ascii=False, indent=2)
        return {"estado": "guardado", "texto": texto}

    def recuperar(self, dueno_id: str, consulta: str, k: int = 3) -> list:
        propios = [r for r in self._recuerdos if r["dueno_id"] == dueno_id]
        if not propios:
            return []
        resultados = IndiceTfidf(propios).buscar(consulta, top_k=k, umbral=0.05)
        if resultados:
            return [d["texto"] for d, _ in resultados]
        # sin coincidencia semántica: devolver el recuerdo más reciente como contexto mínimo
        return [propios[-1]["texto"]]
