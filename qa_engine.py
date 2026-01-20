import os
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ===============================
# Base directory (IMPORTANT)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHUNKS_PATH = os.path.join(BASE_DIR, "data", "vectors", "chunks_store.txt")
INDEX_PATH = os.path.join(BASE_DIR, "data", "vectors", "index.faiss")

# ===============================
# Load embedding model (offline)
# ===============================
model = SentenceTransformer("all-MiniLM-L6-v2")

# ===============================
# Global memory (lazy loaded)
# ===============================
chunks = None
index = None

# ===============================
# Load FAISS + chunks safely
# ===============================
def load_index():
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
def is_incomplete_question(q):
    return len(q.strip().split()) < 3

def extract_best_sentence(chunk, question):
    sentences = re.split(r"(?<=[.!?])\s+", chunk)
    q_words = set(question.lower().split())

    best_sentence = ""
    best_score = 0

    for s in sentences:
        score = len(q_words & set(s.lower().split()))
        if score > best_score:
            best_score = score
            best_sentence = s

    return best_sentence.strip()

# ===============================
# Retrieve best lecture chunk
# ===============================
def find_best_chunk(question):
    load_index()

    if index is None or not chunks:
        return None

    q_embedding = model.encode([question])
    _, indices = index.search(np.array(q_embedding), k=1)

    idx = int(indices[0][0])
    if idx < 0 or idx >= len(chunks):
        return chunks[0]

    return chunks[idx]

# ===============================
# MAIN ANSWER FUNCTION
# ===============================
def answer_question(question):
    question = question.strip()

    # 1️⃣ Incomplete question
    if is_incomplete_question(question):
        return {
            "short": "This question is incomplete. Please ask a complete question.",
            "full": ""
        }

    # 2️⃣ Retrieve lecture memory
    best_chunk = find_best_chunk(question)

    if not best_chunk:
        return {
            "short": "No lecture data found. Please process a lecture first.",
            "full": ""
        }

    # 3️⃣ Extract best sentence
    short_answer = extract_best_sentence(best_chunk, question)

    if not short_answer:
        return {
            "short": "This topic is not covered in the lecture.",
            "full": ""
        }

    # 4️⃣ Final structured response
    return {
        "short": short_answer,
        "full": best_chunk
    }
