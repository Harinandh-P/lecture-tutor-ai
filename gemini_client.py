import os
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError

# Load environment variables
load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ FREE-TIER SAFE MODEL
MODEL_NAME = "models/gemini-flash-latest"

def ask_gemini(question: str, context: str) -> str:
    prompt = f"""
You are an AI tutor.
Answer the question using ONLY the lecture context below.
If the answer is not in the context, say:
"This topic is not covered in the lecture."

Lecture Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text.strip()

    except ClientError as e:
        # 🛡️ Graceful quota handling
        if "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ AI quota limit reached. Please try again in a minute."
        return "⚠️ AI service temporarily unavailable."
