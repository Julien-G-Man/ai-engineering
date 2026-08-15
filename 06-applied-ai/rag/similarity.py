"""
Similarity using Scipy
"""

import numpy as np
from scipy.spatial import distance
from embeddings import create_embeddings, articles

search_text = "machine learning"
search_embedding = create_embeddings(search_text)[0]

distances = []
for article in articles:
    dist = distance.cosine(search_embedding, article["embedding"])
    distances.append(dist)

min_dist_ind = np.argmin(distances)

if __name__ == "__main__":
    print(articles[min_dist_ind]["headline"])