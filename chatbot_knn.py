from config import DOCUMENTS, ANSWERS, MODEL_NAME, K_NEIGHBORS, THRESHOLD
from embeddings import load_embedding_model, build_embeddings
from index import build_knn_index
from retrieval import retrieve_best_match, distance_to_similarity


class KNNEmbeddingChatbot:
    def __init__(self, documents, answers, model_name=MODEL_NAME, k_neighbors=K_NEIGHBORS, threshold=THRESHOLD):
        self.documents = documents
        self.answers = answers
        self.model = load_embedding_model(model_name)
        self.doc_embeddings = build_embeddings(self.model, self.documents)
        self.knn = build_knn_index(self.doc_embeddings, n_neighbors=k_neighbors)
        self.k_neighbors = k_neighbors
        self.threshold = threshold

    def get_best_answer(self, user_question: str) -> str:
        best_distance, best_index = retrieve_best_match(
            user_question, self.model, self.knn, n_neighbors=self.k_neighbors
        )
        similarity = distance_to_similarity(best_distance)
        if best_distance > self.threshold:
            return f"Ich bin mir nicht sicher. Ähnlichkeit nur {similarity:.2f}. Kannst du die Frage anders formulieren?"
        return self.answers[best_index]


def build_chatbot() -> KNNEmbeddingChatbot:
    return KNNEmbeddingChatbot(DOCUMENTS, ANSWERS)
