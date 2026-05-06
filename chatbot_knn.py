# pip install sentence-transformers scikit-learn

from sentence_transformers import SentenceTransformer  # embedding model
from sklearn.neighbors import NearestNeighbors
import numpy as np

# 1. Knowledge base: questions + answers (your context)
documents = [
    "How can I reset my password?",
    "What are your shipping options within Germany?",
    "How do I return a product?",
    "What payment methods do you accept?",
    "How can I contact customer support?"
]

answers = [
    "To reset your password, click “Forgot password” on the login page and follow the instructions.",
    "We ship within Germany using DHL and UPS. Standard shipping takes 2–3 business days.",
    "You can return products within 30 days. Please use the return label included in your package.",
    "We accept credit cards, PayPal, and SEPA bank transfer.",
    "You can contact customer support via email at support@example.com or call +49 123 456 789."
]

# 2. Create embeddings for all documents
model = SentenceTransformer("all-MiniLM-L6-v2")  
doc_embeddings = model.encode(documents, convert_to_numpy=True)

# 3. Fit a kNN index on the embeddings
k = 3
knn = NearestNeighbors(
    n_neighbors=k,
    metric="cosine"
)
knn.fit(doc_embeddings)

def get_best_answer(user_question: str, threshold: float = 0.3) -> str:
    """
    For a user question, find the most similar document via kNN over embeddings
    and return the corresponding answer.
    threshold is a max cosine distance (1 - similarity); smaller = stricter.
    """
    # 4. Embed the incoming question
    q_emb = model.encode([user_question], convert_to_numpy=True)

    # 5. Run kNN search
    distances, indices = knn.kneighbors(q_emb, n_neighbors=k)

    best_distance = float(distances[0][0])
    best_index = int(indices[0][0])

    similarity = 1.0 - best_distance  # since we used cosine *distance*
    if best_distance > threshold:
        return f"Ich bin mir nicht sicher. Ähnlichkeit nur {similarity:.2f}. Kannst du die Frage anders formulieren?"

    return answers[best_index]

# 6. Simple chat loop
if __name__ == "__main__":
    print("Simple kNN-Embedding-Chatbot. Type 'exit' to quit.")
    while True:
        user_inp = input("You: ").strip()
        if user_inp.lower() in {"exit", "quit"}:
            break
        reply = get_best_answer(user_inp)
        print("Bot:", reply)
