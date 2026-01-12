import streamlit as st
import subprocess
import sys
import os

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Lecture Tutor AI Assistant",
    layout="wide"
)

# ===============================
# CSS
# ===============================
st.markdown("""
<style>
.stApp { background-color: #f6f7fb; }

.sticky-header {
    position: sticky;
    top: 0;
    z-index: 999;
    background: linear-gradient(90deg, #6a5acd, #7b68ee);
    padding: 18px 24px;
    color: white;
    font-size: 24px;
    font-weight: 600;
    border-radius: 0 0 16px 16px;
}

.chat-area {
    max-width: 900px;
    margin: auto;
    padding-top: 20px;
}

.user-bubble {
    background-color: #6a5acd;
    color: white;
    padding: 12px 16px;
    border-radius: 18px;
    margin: 10px 0 10px auto;
    max-width: 75%;
}

.ai-bubble {
    background-color: white;
    padding: 14px 18px;
    border-radius: 18px;
    margin: 10px auto 10px 0;
    max-width: 75%;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}

.system-bubble {
    background-color: #eaeaf5;
    color: #333;
    padding: 10px 14px;
    border-radius: 12px;
    margin: 10px auto;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# Header
# ===============================
st.markdown(
    '<div class="sticky-header">✨ Lecture Tutor AI Assistant</div>',
    unsafe_allow_html=True
)

# ===============================
# Folders
# ===============================
os.makedirs("data/audio", exist_ok=True)
os.makedirs("data/transcripts", exist_ok=True)
os.makedirs("data/vectors", exist_ok=True)

# ===============================
# Session state
# ===============================
if "processing" not in st.session_state:
    st.session_state.processing = False

if "process_completed" not in st.session_state:
    st.session_state.process_completed = False

if "memory_ready" not in st.session_state:
    st.session_state.memory_ready = os.path.exists("data/vectors/index.faiss")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ===============================
# Helper
# ===============================
def run_script(script):
    p = subprocess.run([sys.executable, script], capture_output=True, text=True)
    return p.returncode == 0, p.stderr

# ===============================
# Upload + Process
# ===============================
if not st.session_state.process_completed:

    st.subheader("📤 Upload / Rebuild Lecture")

    uploaded_file = st.file_uploader(
        "Upload lecture audio (mp3 / wav / m4a)",
        type=["mp3", "wav", "m4a"],
        disabled=st.session_state.processing
    )

    if uploaded_file:
        with open("data/audio/lecture.mp3", "wb") as f:
            f.write(uploaded_file.read())
        st.info("Lecture uploaded. Click Process Lecture.")

    label = "⏳ Processing..." if st.session_state.processing else "🚀 Process Lecture"

    if st.button(label, disabled=st.session_state.processing):
        st.session_state.processing = True
        st.rerun()

# ===============================
# Processing pipeline
# ===============================
if st.session_state.processing:
    progress = st.progress(0)
    status = st.empty()

    status.write("🎧 Transcribing lecture...")
    ok, err = run_script("transcribe.py")
    if not ok:
        st.error(err)
        st.stop()
    progress.progress(33)

    status.write("✂️ Creating chunks...")
    ok, err = run_script("chunker.py")
    if not ok:
        st.error(err)
        st.stop()
    progress.progress(66)

    status.write("🧠 Building AI memory...")
    ok, err = run_script("embed_store.py")
    if not ok:
        st.error(err)
        st.stop()
    progress.progress(100)

    status.success("🎉 Lecture processed successfully!")

    st.session_state.processing = False
    st.session_state.process_completed = True
    st.session_state.memory_ready = True
    st.rerun()

# ===============================
# Chat UI
# ===============================
from qa_engine import answer_question

st.markdown('<div class="chat-area">', unsafe_allow_html=True)

for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True
        )

    elif msg["role"] == "assistant":

        text = msg["content"]
        if "From Lecture (Context):" in text:
            short, full = text.split("From Lecture (Context):", 1)
        else:
            short, full = text, ""

        st.markdown(
            f"""
            <div class="ai-bubble">
                <div style="font-size:17px;font-weight:700;margin-bottom:6px;">
                    {short.strip()}
                </div>
                <div style="font-size:14px;font-weight:400;line-height:1.5;color:#444;">
                    {"From Lecture (Context): " + full.strip() if full else ""}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f'<div class="system-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Chat input
# ===============================
user_input = st.chat_input("Ask a question based on the lecture...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    if not st.session_state.memory_ready:
        st.session_state.messages.append({
            "role": "system",
            "content": "No lecture memory found. Please process a lecture."
        })
    else:
        res = answer_question(user_input)
        reply = res["short"]
        if res["full"]:
            reply += " From Lecture (Context): " + res["full"]

        st.session_state.messages.append({
            "role": "assistant",
            "content": reply
        })

    st.rerun()
