from sklearn.neighbors import NearestNeighbors
import numpy as np


def build_knn_index(doc_embeddings: np.ndarray, n_neighbors: int = 3) -> NearestNeighbors:
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    knn.fit(doc_embeddings)
    return knn
