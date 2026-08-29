from engine import rag_engine, llm

query = "I want to build my own Jarvis from IronMan"

documents, sources = rag_engine.retrieve(query, top_k=3)
context_prompt = llm.build_prompt_with_context(query, documents)

response = llm.generate_response(context_prompt, sources)
print(response)

