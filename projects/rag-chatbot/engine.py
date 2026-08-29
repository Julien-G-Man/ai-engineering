import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_NAME = 'semantic-search'
NAMESPACE = "youtube-rag-dataset"

class RAGEngine:
    def __init__(self):
        existing = [idx["name"] for idx in pc.list_indexes()]
        if INDEX_NAME not in existing:
            self.create_index(INDEX_NAME)
            
        self.index = pc.Index(INDEX_NAME)
        
    def create_embeddings(self, query: str):
        return client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        
    def retrieve(self, query: str, top_k: int):
        query_emb = self.create_embeddings(query).data[0].embedding
        
        retreived_docs = []
        sources = []
        docs = self.index.query(
            vector=query_emb, 
            top_k=top_k,
            namespace=NAMESPACE,
            include_metadata=True
        )
        
        for doc in docs['matches']:
            retreived_docs.append(doc["metadata"]["text"])
            sources.append((doc["metadata"]["title"], doc["metadata"]["url"]))
            
        return retreived_docs, sources
    
    def upsert(self, vectors: list[float]):
        self.index.upsert(vectors=vectors, namespace=NAMESPACE)
    
    
class LLM():
    def __init__(self):
        self.system_prompt = """You are a helpful assistant that always answers questions"""
        
    def build_prompt_with_context(self, query: str, docs: list):
        delim = '\n\n----\n\n'
        prompt_starter = "Answer the question  based on the following context below.\n\nContext:\n"
        prompt_end = f"\n\nQuestion: {query}\nAnswer: "
        prompt = prompt_starter + delim.join(docs) + prompt_end
        return prompt
    
    def generate_response(self, prompt: str, sources: list):
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        answer = res.choices[0].message.content.strip()
        answer += "\n\nSources:"
        for source in sources:
            answer += "\n" + source[0] + ": " + source[1]
            
        return answer
    
    
llm = LLM()
rag_engine = RAGEngine()