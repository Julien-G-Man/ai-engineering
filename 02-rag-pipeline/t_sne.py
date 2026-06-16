"""
Implementing t-SNE
t-distribution Stochastic Neighbour Embedding
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from embeddings import articles

embeddings = [article["embedding"] for article in articles]

def visualize():
    tsne = TSNE(n_components=2, perplexity=5)
    embeddings_2d = tsne.fit_transform(np.array(embeddings))
    # n_components: the resulting number of dimensions
    # perplexity: used by the algorithm, must be < number of data points

    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

    topics = [article["topic"] for article in articles]

    for i, topic in enumerate(topics):
        plt.annotate(topic, (embeddings_2d[i, 0], embeddings_2d[0, 1]))
          
    plt.show()
    
    
print(visualize())