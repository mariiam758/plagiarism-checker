import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Get current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths relative to script directory
index_path = os.path.join(SCRIPT_DIR, "embeddings", "faiss_index.index")
docs_path = os.path.join(SCRIPT_DIR, "embeddings", "doc_texts.pkl")
output_dir = os.path.join(SCRIPT_DIR, "data")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "user_query_matches.json")

model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"Loading FAISS index from {index_path}")
index = faiss.read_index(index_path)

print(f"Loading documents from {docs_path}")
with open(docs_path, "rb") as f:
    doc_texts = pickle.load(f)

query = input("Enter your sentence or paragraph: ")
query_embedding = model.encode([query]).astype("float32")

k = 7
distances, indices = index.search(query_embedding, k)

print("\nTop Matches:")
results = []
for i, idx in enumerate(indices[0]):
    similarity = 1 - distances[0][i]  # assuming L2 distance normalized
    text = doc_texts[idx]
    print(f"\nMatch #{i+1} - Score: {similarity:.4f}")
    print(text)
    results.append({"score": float(similarity), "text": text})

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved results to {output_path}")
