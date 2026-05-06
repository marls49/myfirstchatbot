import numpy as np


def retrieve_best_match(question: str, model, knn, n_neighbors: int = 3):
    q_emb = model.encode([question], convert_to_numpy=True)
    distances, indices = knn.kneighbors(q_emb, n_neighbors=n_neighbors)
    best_distance = float(distances[0][0])
    best_index = int(indices[0][0])
    return best_distance, best_index


def distance_to_similarity(distance: float) -> float:
    return 1.0 - distance
