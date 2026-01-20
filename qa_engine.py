import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import os

# ===============================
# Load embedding model (offline after first download)
# ===============================
model = SentenceTransformer("all-MiniLM-L6-v2")

# ===============================
# Load chunks + FAISS index safely
# ===============================
CHUNKS_PATH = "data/vectors/chunks_store.txt"
INDEX_PATH = "data/vectors/index.faiss"

chunks = []
index = None

if os.path.exists(CHUNKS_PATH) and os.path.exists(INDEX_PATH):
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = [c.strip() for c in f.read().split("\n\n") if c.strip()]
    index = faiss.read_index(INDEX_PATH)

# ===============================
# Helper: incomplete question
# ===============================
def is_incomplete_question(q):
    q = q.strip().lower()
    return len(q.split()) < 3

# ===============================
# Helper: retrieve best chunk
# ===============================
def find_best_chunk(question):
    if index is None or not chunks:
        return None

    q_embedding = model.encode([question])
    _, indices = index.search(np.array(q_embedding), k=1)

    idx = int(indices[0][0])

    # Safety fallback
    if idx < 0 or idx >= len(chunks):
        return chunks[0]

    return chunks[idx]

# ===============================
# Helper: extract best sentence
# ===============================
def extract_best_sentence(chunk, question):
    sentences = re.split(r"(?<=[.!?])\s+", chunk)
    q_words = set(question.lower().split())

    best_sentence = ""
    best_score = 0

    for s in sentences:
        score = len(q_words.intersection(set(s.lower().split())))
        if score > best_score:
            best_score = score
            best_sentence = s

    return best_sentence.strip()

# ===============================
# Helper: humanize answer (NO AI)
# ===============================
def humanize_answer(sentence):
    if not sentence:
        return ""

    starters = [
        "Simply put, ",
        "In simple terms, ",
        "From the lecture, we can understand that ",
        "The lecture explains that "
    ]

    sentence = sentence.strip()
    sentence = sentence[0].lower() + sentence[1:] if sentence else sentence

    return starters[0] + sentence

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

    # 2️⃣ Retrieve lecture chunk
    best_chunk = find_best_chunk(question)

    if not best_chunk:
        return {
            "short": "No lecture data found. Please process a lecture first.",
            "full": ""
        }

    # 3️⃣ Extract best sentence
    best_sentence = extract_best_sentence(best_chunk, question)

    if not best_sentence:
        return {
            "short": "This topic is not covered in the lecture.",
            "full": ""
        }

    # 4️⃣ Humanized short answer
    short_answer = humanize_answer(best_sentence)

    # 5️⃣ Final response
    return {
        "short": short_answer,
        "full": best_chunk
    }
