from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading embedding model...")

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Reading documents...")

# Read the sentences
with open("data/sentences.txt", "r") as f:
    documents = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(documents)} documents.")

# Create embeddings
document_embeddings = model.encode(documents)

print("Embeddings created!")

query = input("\nAsk a question: ")

query_embedding = model.encode(query)

similarities = np.dot(document_embeddings, query_embedding) / (
    np.linalg.norm(document_embeddings, axis=1) *
    np.linalg.norm(query_embedding)
)

top_indices = np.argsort(similarities)[::-1][:3]

print("\nTop 3 Results:\n")

for idx in top_indices:
    print(f"Similarity: {similarities[idx]:.4f}")
    print(documents[idx])
    print("-" * 50)