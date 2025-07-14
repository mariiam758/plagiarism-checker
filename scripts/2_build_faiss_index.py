import os
import numpy as np
import faiss

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to script folder
embeddings_path = os.path.join(SCRIPT_DIR, "embeddings", "doc_embeddings.npy")
index_path = os.path.join(SCRIPT_DIR, "embeddings", "faiss_index.index")

# Load embeddings
embeddings = np.load(embeddings_path).astype("float32")

# Create FAISS index (L2 distance)
index = faiss.IndexFlatL2(embeddings.shape[1])

# Add embeddings to index
index.add(embeddings)

# Save index
faiss.write_index(index, index_path)
print(f"FAISS index built and saved to {index_path}")
