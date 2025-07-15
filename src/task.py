from api.utils import call_api
from langchain.agents import tool

@tool(return_direct=True)
def sentiment_analysis(text: str) -> str:
    """Analyse le sentiment (positif, négatif, neutre)"""
    return call_api("sentiment_analysis", {"text": text})

@tool(return_direct=True)
def emotion_analysis(text: str) -> str:
    """Analyse les émotions (joie, colère, tristesse, etc.)"""
    return call_api("emotion_analysis", {"text": text})

@tool(return_direct=True)
def ner_analysis(text: str) -> str:
    """Fait de la reconnaissance d'entités nommées (NER)."""
    return call_api("ner_analysis", {"text": text})