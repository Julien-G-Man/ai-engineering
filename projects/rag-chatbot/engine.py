import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from repo import store

load_dotenv()

INDEX_NAME = 'semantic-search'
NAMESPACE = "youtube-rag-dataset"

class RAGEngine:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        existing = [idx["name"] for idx in self.pc.list_indexes()]
        if INDEX_NAME not in existing:
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=1536,
                metric = 'dotproduct', # can also be cosine or euclidean
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        self.index = self.pc.Index(INDEX_NAME)
        
    def create_embeddings(self, query: str):
        return self.openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        
    def retrieve(self, query: str, top_k: int) -> tuple[list[str], list[tuple[str, str]]]:
        query_emb = self.create_embeddings(query).data[0].embedding
        retrieved_docs = []
        sources = []
        docs = self.index.query(
            vector=query_emb, 
            top_k=top_k,
            namespace=NAMESPACE,
            include_metadata=True
        )
        for doc in docs['matches']:
            retrieved_docs.append(doc["metadata"]["text"])
            sources.append((doc["metadata"]["title"], doc["metadata"]["url"]))
        return retrieved_docs, sources
    
    def upsert(self, vectors):
        self.index.upsert(vectors=vectors, namespace=NAMESPACE)
    
    
class LLM():
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = """You are a helpful assistant that always answers questions. Keep it very brief."""
        
    def build_prompt_with_context(self, query: str, docs: list):
        delim = '\n\n----\n\n'
        prompt_starter = "Answer the question  based on the following context below.\n\nContext:\n"
        prompt_end = f"\n\nQuestion: {query}\nAnswer: "
        prompt = prompt_starter + delim.join(docs) + prompt_end
        return prompt
    
    
    def generate_response(self, prompt: str, sources: list[tuple[str, str]]) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        
        for msg in store.get_past_messages():
            messages.append({"role": "user", "content": msg["user"]})
            messages.append({"role": "assistant", "content": msg["ai"]})
            
        messages.append({"role": "user", "content": prompt})
        
        res = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        answer = res.choices[0].message.content.strip()
        answer += "\n\nSources:"
        for source in sources:
            answer += "\n" + source[0] + ": " + source[1]
        return answer
    
    
    
