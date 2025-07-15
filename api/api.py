from api.utils import load_models, ner_ncbi_pipeline, emotion_pipeline, Sentiment_pipeline
from fastapi import FastAPI, Request

app = FastAPI(title="Emotion Detection API", version="1.0")
sentiment_pipeline = Sentiment_pipeline()
MODELS, TOKENIZER = load_models()

# --------- FastAPI ----------
@app.post("/emotion_analysis")
async def root(request: Request):
    data = await request.json()
    text = data["text"]
    emotion_bert_lora = MODELS["Emotion"]
    pred = emotion_pipeline(text, emotion_bert_lora, TOKENIZER)
    return {"result":  pred}

@app.post("/sentiment_analysis")
async def root(request: Request):
    data = await request.json()
    text = data["text"]
    sentiment_analysis_model = MODELS["Sentiment"]
    pred = sentiment_pipeline.pipeline(text, sentiment_analysis_model)
    return {"result":  pred}

@app.post("/ner_analysis")
async def root(request: Request):
    data = await request.json()
    text = data["text"]
    ner_ncbi_bert_lora = MODELS["Ner"]
    pred = ner_ncbi_pipeline(ner_ncbi_bert_lora, text)
    return {"result": pred}


