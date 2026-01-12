import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ===============================
# Paths (STANDARDIZED)
# ===============================
CHUNKS_PATH = "data/transcripts/chunks.txt"
INDEX_PATH = "data/vectors/index.faiss"
CHUNK_STORE_PATH = "data/vectors/chunks_store.txt"

# ===============================
# Load embedding model
# ===============================
model = SentenceTransformer("all-MiniLM-L6-v2")

# ===============================
# Load chunks safely
# ===============================
if not os.path.exists(CHUNKS_PATH):
    raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# ===============================
# Parse chunks (CHUNK N format)
# ===============================
raw_chunks = content.split("CHUNK")
chunks = []

for block in raw_chunks:
    block = block.strip()
    if not block:
        continue

    lines = block.split("\n", 1)
    if len(lines) == 2:
        text = lines[1].strip()
        if len(text.split()) >= 10:   # safety filter
            chunks.append(text)

if not chunks:
    raise ValueError("No valid chunks found to embed.")

print(f"Total chunks loaded: {len(chunks)}")

# ===============================
# Create embeddings
# ===============================
embeddings = model.encode(chunks, show_progress_bar=True)

# ===============================
# Build FAISS index
# ===============================
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# ===============================
# Save FAISS index
# ===============================
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
faiss.write_index(index, INDEX_PATH)

# ===============================
# Save chunks for retrieval
# ===============================
with open(CHUNK_STORE_PATH, "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk.strip() + "\n\n")

print("Embeddings stored successfully.")
