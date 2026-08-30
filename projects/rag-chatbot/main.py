from fastapi import FastAPI, HTTPException, Depends
from engine import RAGEngine, LLM
from schemas import ChatQuery, ChatResponse, EmbeddingQuery, EmbeddingResponse
from repo import store

app = FastAPI()

llm = None
rag_engine = None

def get_llm() -> LLM:
    global llm
    if llm is None:
        llm = LLM()
    return llm

def get_rag_engine() -> RAGEngine:
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine()
    return rag_engine


@app.get("/")
def root():
    return {"message": "RAG chatbot API"}


@app.post("/generate")
def generate_answer(
    query: ChatQuery, 
    rag_engine: RAGEngine = Depends(get_rag_engine), 
    llm: LLM = Depends(get_llm)
    ):
    try:
        documents, sources = rag_engine.retrieve(query.text, top_k=3)
        context_prompt = llm.build_prompt_with_context(query.text, documents)
        response = llm.generate_response(context_prompt, sources)
        store.save_message(query.text, response)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed")
def embed(query: EmbeddingQuery, rag_engine = Depends(get_rag_engine)):
    try:
        response  = rag_engine.create_embeddings(query.text)
        embedding = response.data[0].embedding
        return EmbeddingResponse(embedding=embedding)
    except ValueError:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
        
@app.get("/past_msg")
def get_past_messages():
    past_msg = store.get_past_messages()
    return {"past_msg" : past_msg}