from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model (offline after first download)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read chunks
chunks_path = "data/transcripts/chunks.txt"
with open(chunks_path, "r", encoding="utf-8") as f:
    content = f.read()

# Split chunks by marker
raw_chunks = content.split("CHUNK")
chunks = []

for chunk in raw_chunks:
    chunk = chunk.strip()
    if chunk:
        lines = chunk.split("\n", 1)
        if len(lines) == 2:
            chunks.append(lines[1].strip())

print(f"Total chunks loaded: {len(chunks)}")

# Convert chunks to embeddings
embeddings = model.encode(chunks)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index
faiss.write_index(index, "data/vectors/index.faiss")

# Save chunks separately (for later retrieval)
with open("data/vectors/chunks_store.txt", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk + "\n---\n")

print("Embeddings stored successfully.")
