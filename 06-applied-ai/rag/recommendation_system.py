import numpy as np
from embeddings import articles, create_embeddings, cosine_similarity
from semantic_search import create_article_text, find_n_closest

user_history = [
    {"headline": "How Nvidia GPUs could decide Who Wins the AI Race",
    "topic": "Tech",
    "keywords": ["ai", "business", "computers"]},
    {"headline": "Tech Giant Buys 49% Stake In AI Startup",
     "topic": "Tech",
     "keywords": ["business", "AI", "acquisition"]}
]

current_article = {
    "headline": "African tech prodigee accepted into the Bank of America Technology Internship",
    "topic": "Technology",
    "keywords": ["tech talent", "software engineer", "banktech", "fintech", "internship", "student"]
}


def recommend():
    article_texts = [create_article_text(article) for article in articles]
    current_article_text = create_article_text(current_article)
    print(current_article_text + "\n")

    current_article_embeddings = create_embeddings(current_article_text)[0]
    article_embeddings = create_embeddings(article_texts)

    hits = find_n_closest(current_article_embeddings, article_embeddings)

    print("Hits...")
    display_hits(hits, articles)
    
    

def recommend_on_multiple_data_points():
    history_texts = [create_article_text(article) for article in user_history]
    history_embeddings = create_embeddings(history_texts)
    mean_history_embeddings = np.mean(history_embeddings, axis=0)
    
    articles_filtered = [article for article in articles if article not in user_history]
    article_texts = [create_article_text(article) for article in articles_filtered]
    article_embeddings = create_embeddings(article_texts)
    
    hits = find_n_closest(mean_history_embeddings, article_embeddings)
    
    print("\nFiltered Hits According to history...")
    display_hits(hits, articles)
   

def display_hits(hits, articles):
    for hit in hits:
        article = articles[hit['index']]
        print(f"{hits.index(hit) + 1}: ", article['headline'])        
        

if __name__ == "__main__":
    recommend()
    recommend_on_multiple_data_points()