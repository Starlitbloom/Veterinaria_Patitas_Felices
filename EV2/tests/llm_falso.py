"""LLM guionado para probar la ORQUESTACIÓN sin costo ni red.

Consume una lista de respuestas en orden (primero la del planificador, luego las
del agente). Una respuesta puede ser un AIMessage o una función(mensajes)->AIMessage
para reaccionar al resultado de una tool, como haría un LLM real.
NO valida la calidad del LLM; valida que el grafo, las tools y la memoria funcionan.
"""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class LLMGuionado(BaseChatModel):
    respuestas: list
    idx: int = 0
    vistos: list = []  # lista de listas de mensajes que recibió cada llamada

    @property
    def _llm_type(self):
        return "guionado"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.vistos.append(list(messages))
        r = self.respuestas[self.idx]
        self.idx += 1
        if callable(r):
            r = r(messages)
        return ChatResult(generations=[ChatGeneration(message=r)])


def llamar(nombre, _id="c1", **args):
    return AIMessage(content="", tool_calls=[{"name": nombre, "args": args, "id": _id}])


def texto(t):
    return AIMessage(content=t)
