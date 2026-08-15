"""
Sample RAG pipeline in pure Python - No vector databases
"""

import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

lecture_notes = """
Artificial Intelligence agents use tool calling to execute external actions. 
FastAPI is a modern, fast web framework for building APIs with Python 3.8+.
Django handles user authentication, session security, and relational databases seamlessly.
Cosine similarity calculates the cosine of the angle between two multi-dimensional vectors.
"""

# Simple Semantic Chunking
# In a production app, we can split by paragraphs or custom semantic boundaries
chunks = [chunk.strip() for chunk in lecture_notes.strip().split("\n")]

# Generate Embeddings using NumPy
def get_embedding(text: str, model="text-embedding-3-small") -> np.ndarray:
    """Fetches embedding vector and returns it as a normalized NumPy array."""
    response = client.embeddings.create(input=[text], model=model)
    vector = response.data[0].embedding
    return np.array(vector, dtype=np.float32)

print("Encoding textbook chunks into vectors...")
# Create a matrix of all our textbook chunk embeddings
chunk_embeddings = np.array([get_embedding(c) for c in chunks])

# Pure NumPy Vector Search (Replaces Vector DBs like Pinecone for small/mid scales)
def search_top_k(query: str, chunk_vectors: np.ndarray, text_chunks: list, k=1):
    """Calculates dot product similarity and returns the best matching context."""
    query_vector = get_embedding(query)
    
    # Mathematical Dot Product matrix multiplication (Calculates similarity for ALL chunks at once)
    similarities = np.dot(chunk_vectors, query_vector)
    
    # Get indices sorted from highest similarity to lowest
    top_indices = np.argsort(similarities)[::-1][:k]
    
    return [(text_chunks[idx], similarities[idx]) for idx in top_indices]


def get_llm_response(system_prompt, user_prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )
    return completion.choices[0].message.content


query = "How do I secure user accounts in my Python backend?"
results = search_top_k(query, chunk_embeddings, chunks, k=1)
best_match_text, similarity_score = results[0]

print(f"\n[Search Match Found! Score: {similarity_score:.4f}]")
print(f"Context: {best_match_text}")


system_prompt = "You are a precise study tutor. Answer the user prompt using ONLY the provided Context."
user_prompt = f"Context: {best_match_text}\n\nQuestion: {query}"
response = get_llm_response(system_prompt, user_prompt)

print("\n[AI Tutor Answer]:")
print(response)
