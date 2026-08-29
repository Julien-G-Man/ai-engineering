import json
from fastapi import FastAPI
from engine import llm, rag_engine

app = FastAPI()

@app.get("/")
def root():
    return json({"message": "RAG chatbot API"})