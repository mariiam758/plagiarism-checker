import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths relative to script
input_path = os.path.join(SCRIPT_DIR, "data", "user_query_matches.json")
output_dir = os.path.join(SCRIPT_DIR, "heatmaps")
os.makedirs(output_dir, exist_ok=True)

# Load similarity results saved from script 3
with open(input_path, "r", encoding="utf-8") as f:
    results = json.load(f)

scores = [item["score"] for item in results]
texts = [item["text"][:80].replace("\n", " ") + "..." for item in results]  # truncate & clean

# Set up figure
sns.set(font_scale=0.6)
fig, ax = plt.subplots(figsize=(8, len(texts) * 0.6 + 2))

# Plot heatmap
sns.heatmap(
    np.array(scores).reshape(-1, 1),
    annot=True,
    yticklabels=texts,
    xticklabels=["Similarity"],
    cmap="YlOrRd",
    linewidths=0.5,
    cbar=False,
    ax=ax
)

plt.title("🔍 User Input Similarity Heatmap", fontsize=12)
plt.tight_layout()

# Save to file (timestamped filename)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join(output_dir, f"query_{timestamp}.png")
plt.savefig(filename, dpi=300)
print(f"✅ Heatmap saved to {filename}")

# Show the plot
plt.show()
