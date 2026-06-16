from scipy.spatial import distance
from embeddings import articles, embed_text


def create_article_text(article):
    return f"""Headline: {article["headline"]}
Topic: {article["topic"]}
Keywords: {',' .join(article["keywords"])}"""


def find_n_closest(query_vector, embeddings, n=3):
    distances = []
    for index, embedding in enumerate(embeddings):
        dist = distance.cosine(query_vector, embedding)
        distances.append({"distance": dist, "index": index})
    distances_sorted = sorted(distances, key=lambda x: x["distance"])
    return distances_sorted[0:n]


def main():
    print("====== Getting closest topics ======")
    article_texts = [create_article_text(article) for article in articles]
    article_embeddings = [item["embedding"] for item in embed_text(article_texts).model_dump()["data"]]

    query_text = "AI"
    query_vector = embed_text(query_text).model_dump()["data"][0]["embedding"]
    hits = find_n_closest(query_vector, article_embeddings)
    
    print(f'\nSearch results for "{query_text}"')
    for hit in hits: 
        article = articles[hit["index"]]
        print(article["headline"])



if __name__ == "__main__":
    print(main())