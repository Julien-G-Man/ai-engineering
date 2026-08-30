from engine import RAGEngine, LLM

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

llm = LLM()
rag_engine = RAGEngine()

def main():
    query = "I want to build my own Jarvis from IronMan"

    documents, sources = rag_engine.retrieve(query, top_k=3)
    context_prompt = llm.build_prompt_with_context(query, documents)

    response = llm.generate_response(context_prompt, sources)
    print(response)
    
if __name__ == "__main__":
    print(main())

