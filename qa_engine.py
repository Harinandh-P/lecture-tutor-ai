import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import os

from gemini_client import ask_gemini   # ✅ NEW

# ===============================
# Load embedding model
# ===============================
model = SentenceTransformer("all-MiniLM-L6-v2")

# ===============================
# Load chunks safely
# ===============================
CHUNKS_PATH = "data/vectors/chunks_store.txt"
INDEX_PATH = "data/vectors/index.faiss"

if not os.path.exists(CHUNKS_PATH) or not os.path.exists(INDEX_PATH):
    chunks = []
    index = None
else:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = [c.strip() for c in f.read().split("\n\n") if c.strip()]
    index = faiss.read_index(INDEX_PATH)

# ===============================
# Helper: detect incomplete question
# ===============================
def is_incomplete_question(q):
    q = q.lower().strip()
    return len(q.split()) < 3

# ===============================
# Safe chunk retrieval
# ===============================
def find_best_chunk(question):
    if index is None or not chunks:
        return None

    q_embedding = model.encode([question])
    _, indices = index.search(np.array(q_embedding), k=1)

    idx = int(indices[0][0])

    # 🔐 Safety check
    if idx < 0 or idx >= len(chunks):
        return chunks[0]

    return chunks[idx]

# ===============================
# MAIN ANSWER FUNCTION (FINAL)
# ===============================
def answer_question(question):
    question = question.strip()

    # 1️⃣ Incomplete question check
    if is_incomplete_question(question):
        return {
            "short": "This question is incomplete. Please ask a complete question.",
            "full": ""
        }

    # 2️⃣ Retrieve best lecture chunk
    best_chunk = find_best_chunk(question)

    if not best_chunk:
        return {
            "short": "No lecture data found. Please process a lecture first.",
            "full": ""
        }

    # 3️⃣ Ask Gemini using lecture-only context
    answer = ask_gemini(question, best_chunk)

    # 4️⃣ Safety: Gemini fallback
    if not answer or "not covered" in answer.lower():
        return {
            "short": "This topic is not covered in the lecture.",
            "full": ""
        }

    # 5️⃣ Return structured response
    return {
        "short": answer.split("\n")[0].strip(),
        "full": best_chunk
    }
