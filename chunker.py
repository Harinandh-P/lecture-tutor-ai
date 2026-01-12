import os
import re

# ===============================
# Paths (STANDARDIZED)
# ===============================
TRANSCRIPT_PATH = "data/transcripts/lecture.txt"
OUTPUT_PATH = "data/transcripts/chunks.txt"   # ✅ FIXED PATH

# ===============================
# Chunk configuration
# ===============================
CHUNK_WORD_LIMIT = 70
OVERLAP_SENTENCES = 1
MIN_CHUNK_WORDS = 20   # discard tiny chunks

# ===============================
# Load transcript
# ===============================
def load_transcript():
    if not os.path.exists(TRANSCRIPT_PATH):
        raise FileNotFoundError(f"Transcript not found: {TRANSCRIPT_PATH}")

    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

# ===============================
# Sentence split
# ===============================
def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)

# ===============================
# Chunk creation
# ===============================
def create_chunks(sentences):
    raw_chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        current_chunk.append(sentence)
        current_word_count += len(words)

        if current_word_count >= CHUNK_WORD_LIMIT:
            raw_chunks.append(" ".join(current_chunk))

            # overlap last sentence
            current_chunk = current_chunk[-OVERLAP_SENTENCES:]
            current_word_count = sum(len(s.split()) for s in current_chunk)

    if current_chunk:
        raw_chunks.append(" ".join(current_chunk))

    return raw_chunks

# ===============================
# Clean & validate chunks
# ===============================
def clean_chunks(chunks):
    clean = []
    seen = set()

    for c in chunks:
        c = c.strip()
        word_count = len(c.split())

        if word_count < MIN_CHUNK_WORDS:
            continue
        if c in seen:
            continue

        seen.add(c)
        clean.append(c)

    return clean

# ===============================
# Save chunks
# ===============================
def save_chunks(chunks):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"CHUNK {i}\n{chunk}\n\n")

# ===============================
# Main
# ===============================
def main():
    text = load_transcript()
    sentences = split_sentences(text)

    raw_chunks = create_chunks(sentences)
    final_chunks = clean_chunks(raw_chunks)

    save_chunks(final_chunks)

    print(f"Raw chunks created   : {len(raw_chunks)}")
    print(f"Final chunks written : {len(final_chunks)}")

# ===============================
# Entry
# ===============================
if __name__ == "__main__":
    main()
