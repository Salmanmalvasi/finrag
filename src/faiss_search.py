from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read documents
with open("data/sentences.txt", "r") as f:
    documents = [line.strip() for line in f if line.strip()]

# Create embeddings
embeddings = model.encode(documents)

# FAISS expects float32
embeddings = np.array(embeddings).astype("float32")

# Dimension of each embedding
dimension = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)

# Add embeddings
index.add(embeddings)

print(f"Indexed {index.ntotal} documents.")

# Ask question
query = input("Ask a question: ")

query_embedding = model.encode([query]).astype("float32")

# Search
k = 3

distances, indices = index.search(query_embedding, k)

print("\nTop Results:\n")

for idx in indices[0]:
    print(documents[idx])