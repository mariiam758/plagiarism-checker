import streamlit as st
import os
import pickle
import faiss
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
from sentence_transformers import SentenceTransformer

# === Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(SCRIPT_DIR, "embeddings", "faiss_index.index")
texts_path = os.path.join(SCRIPT_DIR, "embeddings", "doc_texts.pkl")
output_data_dir = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(output_data_dir, exist_ok=True)

# === Load model and data
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(index_path)

with open(texts_path, "rb") as f:
    doc_texts = pickle.load(f)

# === Streamlit UI
st.title("📚 Semantic Plagiarism Checker")
st.markdown("Enter your paragraph to find semantically similar academic content.")

query = st.text_area("📝 Enter your sentence or paragraph", height=150)
top_k = st.slider("🔢 Top matches to retrieve", min_value=3, max_value=10, value=5)

if st.button("🔍 Run Similarity Check"):
    if not query.strip():
        st.warning("Please enter a sentence or paragraph.")
    else:
        # === Embed query and search
        query_embedding = model.encode([query]).astype("float32")
        distances, indices = index.search(query_embedding, top_k)

        # === Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            similarity = 1 - distances[0][i]
            results.append({
                "rank": i + 1,
                "score": float(similarity),
                "text": doc_texts[idx]
            })

        # === Generate unique filename ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        filename_prefix = f"{timestamp}_{query_hash}"

        # === Save JSON with query
        json_path = os.path.join(output_data_dir, f"user_query_matches_{filename_prefix}.json")
        output_payload = {
            "query": query.strip(),
            "top_k": top_k,
            "results": results
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2, ensure_ascii=False)
        st.success(f"✅ Similarity scores saved to `{json_path}`")

        # === Show results
        st.subheader("📊 Top Matches:")
        for r in results:
            st.markdown(f"**#{r['rank']}** — Score: `{r['score']:.4f}`")
            st.write(r["text"])
            st.markdown("---")

        # === Plot heatmap
        st.subheader("🔥 Similarity Heatmap")
        scores = [r["score"] for r in results]
        labels = [r["text"][:60].replace("\n", " ") + "..." for r in results]

        fig, ax = plt.subplots(figsize=(8, len(scores) * 0.6 + 2))
        sns.heatmap(
            np.array(scores).reshape(-1, 1),
            annot=True,
            yticklabels=labels,
            xticklabels=["Similarity"],
            cmap="YlGnBu",
            linewidths=0.5,
            cbar=False,
            ax=ax
        )
        plt.title("Similarity Heatmap", fontsize=12)
        plt.suptitle(f"Query: {query[:80]}{'...' if len(query) > 80 else ''}", fontsize=9, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve space at top for suptitle

        # === Save heatmap image
        heatmap_path = os.path.join(output_data_dir, f"heatmap_{filename_prefix}.png")
        fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        st.pyplot(fig)
        st.success(f"🖼️ Heatmap image saved to `{heatmap_path}`")

        # === Echo query for reference
        st.markdown(f"🧾 **Query Used:** `{query.strip()}`")
