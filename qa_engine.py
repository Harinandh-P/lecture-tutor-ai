import os
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ===============================
# Paths
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "vectors", "chunks_store.txt")
INDEX_PATH = os.path.join(BASE_DIR, "data", "vectors", "index.faiss")

# ===============================
# Model
# ===============================
model = SentenceTransformer("all-MiniLM-L6-v2")

# ===============================
# Memory (lazy)
# ===============================
chunks = None
index = None

# ===============================
# Load index safely
# ===============================
def load_memory():
    global chunks, index
    if chunks is not None and index is not None:
        return

    if not os.path.exists(CHUNKS_PATH) or not os.path.exists(INDEX_PATH):
        chunks = []
        index = None
        return

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = [c.strip() for c in f.read().split("\n\n") if c.strip()]

    index = faiss.read_index(INDEX_PATH)

# ===============================
# Helpers
# ===============================
def is_incomplete(q):
    return len(q.split()) < 3

def extract_best_sentence(chunk, question):
    sentences = re.split(r"(?<=[.!?])\s+", chunk)
    q_words = set(question.lower().split())

    best = ""
    score = 0

    for s in sentences:
        overlap = len(q_words & set(s.lower().split()))
        if overlap > score:
            score = overlap
            best = s

    return best.strip(), score

# ===============================
# MAIN FUNCTION
# ===============================
def answer_question(question):
    question = question.strip()

    # 1️⃣ Incomplete
    if is_incomplete(question):
        return {
            "short": "This question is incomplete. Please ask a complete question.",
            "full": ""
        }

    load_memory()

    if index is None or not chunks:
        return {
            "short": "No lecture data found. Please process a lecture first.",
            "full": ""
        }

    # 2️⃣ Search
    q_emb = model.encode([question])
    distances, indices = index.search(np.array(q_emb), k=1)

    similarity_distance = distances[0][0]
    idx = int(indices[0][0])

    # 🔐 HARD semantic rejection
    if similarity_distance > 1.2:
        return {
            "short": "This topic is not covered in the lecture.",
            "full": ""
        }

    best_chunk = chunks[idx]

    # 3️⃣ Extract answer sentence
    short_answer, overlap_score = extract_best_sentence(best_chunk, question)

    # 🔐 Keyword relevance check
    if overlap_score < 2:
        return {
            "short": "This topic is not covered in the lecture.",
            "full": ""
        }

    # 4️⃣ Final answer
    return {
        "short": short_answer,
        "full": best_chunk
    }
