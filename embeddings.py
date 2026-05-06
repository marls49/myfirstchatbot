from sentence_transformers import SentenceTransformer
import numpy as np


def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def build_embeddings(model: SentenceTransformer, documents: list[str]) -> np.ndarray:
    return model.encode(documents, convert_to_numpy=True)
