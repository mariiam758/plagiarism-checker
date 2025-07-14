import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build paths
data_path = os.path.join(SCRIPT_DIR, "data", "arxiv_abstracts.json")
embeddings_dir = os.path.join(SCRIPT_DIR, "embeddings")
os.makedirs(embeddings_dir, exist_ok=True)

embeddings_path = os.path.join(embeddings_dir, "doc_embeddings.npy")
texts_path = os.path.join(embeddings_dir, "doc_texts.pkl")

# Load dataset
with open(data_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

docs = dataset.get("docs", [])
print(f"Loaded {len(docs)} documents.")

if not docs:
    raise ValueError("No documents found in dataset!")

# Embed
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding dataset...")
embeddings = model.encode(docs, show_progress_bar=True)

# Save
np.save(embeddings_path, embeddings)
with open(texts_path, "wb") as f:
    pickle.dump(docs, f)

print(f"✅ Embeddings saved to: {embeddings_path}")
print(f"✅ Texts saved to: {texts_path}")
