"""
Similarity using Scipy
"""

import numpy as np
from scipy.spatial import distance
from embeddings import embed_text, articles

search_text = "machine learning"
search_embedding = embed_text(search_text).model_dump()["data"][0]["embedding"]

distances = []
for arti in articles:
    dist = distance.cosine(search_embedding, arti["embedding"])
    distances.append(dist)

min_dist_ind = np.argmin(distances)

if __name__ == "__main__":
    print(articles[min_dist_ind]["headline"])