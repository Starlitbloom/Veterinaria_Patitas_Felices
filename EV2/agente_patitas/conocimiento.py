"""Base de conocimiento (RAG) reutilizada de la EP1, con índice TF-IDF.

El mismo IndiceTfidf se usa para el conocimiento de la clínica y para la
memoria de largo plazo (recuperación semántica de contexto).
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS_ES = [
    "el", "la", "los", "las", "de", "del", "en", "y", "a", "que", "qué", "es",
    "un", "una", "unos", "unas", "para", "por", "con", "mi", "mis", "su", "sus",
    "me", "se", "te", "puedo", "puede", "pueden", "cómo", "como", "dónde",
    "donde", "cuál", "cuales", "cual", "más", "si", "no", "le", "les", "lo",
    "al", "o", "sobre", "hay", "está", "estan", "están", "soy", "eres", "ser",
    "tu", "tus", "yo", "nos", "ese", "esa", "este", "esta", "también", "muy",
    "ya", "pero", "porque", "cuando", "tener", "tengo", "tiene",
]

FUENTES = [
    {"id": "faq_agendar_cita", "texto": "¿Cómo agendo una hora para mi mascota? Puedes solicitar tu cita en línea desde el botón 'Solicitar cita', eligiendo el servicio, la fecha y la hora. Nuestro equipo confirma la solicitud a la brevedad."},
    {"id": "faq_especies", "texto": "¿Qué especies atienden? Atendemos perros, gatos, conejos y aves."},
    {"id": "faq_historial_clinico", "texto": "¿Puedo ver el historial clínico de mi mascota en línea? Sí. Una vez que creas tu cuenta, puedes revisar el historial de consultas, vacunas y tratamientos desde tu perfil."},
    {"id": "faq_reprogramar_cita", "texto": "¿Qué pasa si no puedo asistir a mi hora? Puedes reagendar o cancelar tu cita desde 'Mis citas' con anticipación, para que ese horario quede disponible para otra mascota."},
    {"id": "horario_ubicacion", "texto": "Horario de atención y ubicación: Veterinaria Patitas Felices atiende de lunes a viernes de 09:00 a 19:00 hrs. Está ubicada en Rancagua, Región de O'Higgins."},
    {"id": "catalogo_vacunas", "texto": "Catálogo de vacunas: antirrábica canina (VA001), séxtuple canina (VA002), bivalente felina (VA003), triple felina (VA004), Bordetella canina (VA005), antirrábica felina (VA006)."},
    {"id": "guia_sag_vacunacion", "texto": "El Servicio Agrícola y Ganadero (SAG) recomienda mantener al día el esquema de vacunación antirrábica en caninos y felinos como medida de salud pública."},
]


class IndiceTfidf:
    """Índice vectorial en memoria: TF-IDF + similitud coseno."""

    def __init__(self, documentos: list):
        self.documentos = documentos
        self.vectorizer = TfidfVectorizer(stop_words=STOPWORDS_ES)
        self.matriz = self.vectorizer.fit_transform([d["texto"] for d in documentos])

    def buscar(self, consulta: str, top_k: int = 2, umbral: float = 0.08):
        sims = cosine_similarity(self.vectorizer.transform([consulta]), self.matriz)[0]
        orden = sorted(zip(sims, self.documentos), key=lambda x: x[0], reverse=True)
        return [(d, float(s)) for s, d in orden[:top_k] if s >= umbral]


INDICE_CLINICA = IndiceTfidf(FUENTES)
