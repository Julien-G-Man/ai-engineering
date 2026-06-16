import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(text: str | list[str]):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response

def cosine_similarity(a: list[float], b: list[float]) -> float:
    vec_a, vec_b = np.array(a), np.array(b)
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

articles = [
    {
        "headline": "AI startup secures new funding to expand product team",
        "topic": "Technology",
        "keywords": ["ai", "startup", "funding", "product", "team expansion"]
    },
    {
        "headline": "Hospitals adopt new screening program to improve patient care",
        "topic": "Health",
        "keywords": ["hospitals", "screening", "program", "patient care", "health outcomes"]
    },
    {
        "headline": "National team wins dramatic overtime final",
        "topic": "Sports",
        "keywords": ["national team", "overtime", "final", "victory", "championship"]
    },
    {
        "headline": "Researchers uncover a promising clue in climate trends",
        "topic": "Science",
        "keywords": ["researchers", "climate", "trends", "discovery", "evidence"]
    },
    {
        "headline": "Parliament debates tax reform ahead of the next vote",
        "topic": "Politics",
        "keywords": ["parliament", "tax reform", "debate", "vote", "policy"]
    },
    {
        "headline": "Central bank signals possible rate pause after inflation slows",
        "topic": "Finance",
        "keywords": ["central bank", "interest rates", "pause", "inflation", "monetary policy"]
    },
    {
        "headline": "Schools introduce tutoring support for struggling students",
        "topic": "Education",
        "keywords": ["schools", "tutoring", "support", "students", "learning"]
    },
    {
        "headline": "City expands green transit routes to reduce emissions",
        "topic": "Environment",
        "keywords": ["city", "green transit", "routes", "emissions", "sustainability"]
    },
    {
        "headline": "Travel demand rises as airlines add summer routes",
        "topic": "Travel",
        "keywords": ["travel demand", "airlines", "summer routes", "passengers", "tourism"]
    },
    {
        "headline": "Film festival spotlights emerging voices from around the world",
        "topic": "Culture",
        "keywords": ["film festival", "emerging voices", "global", "cinema", "culture"]
    }
]

headline_texts = [t["headline"] for t in articles]

response = embed_text(headline_texts)
response_dict = response.model_dump()

for i, article in enumerate(articles):
    article["embedding"] = response_dict["data"][i]["embedding"]
    
first = articles[0]["embedding"]


def get_similarity_scores():
    for i in range(len(articles)):
        art = articles[i]["embedding"]
        sim = cosine_similarity(first, art)
        print(f"{i+1} - Similarity score: {sim}")



if __name__ == "__main__":
    get_similarity_scores()