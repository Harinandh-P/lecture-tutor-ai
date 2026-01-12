import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Create client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# List available models
models = client.models.list()

print("AVAILABLE MODELS FOR YOUR ACCOUNT:\n")

found = False
for model in models:
    print(model.name)
    found = True

if not found:
    print("❌ No models available for this account.")
