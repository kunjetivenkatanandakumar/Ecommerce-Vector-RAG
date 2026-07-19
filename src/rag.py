from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env from the project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Read API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Check your .env file.")

# Create Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


def ask_gemini(query, context):
    prompt = f"""
You are an AI shopping assistant.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=512,
        ),
    )

    return response.text


if __name__ == "__main__":
    context = """
Product:
Apple iPhone 15

Description:
128GB storage, A16 Bionic chip, Dynamic Island, USB-C charging.
"""

    question = "Which phone is good for photography?"

    answer = ask_gemini(question, context)
    print(answer)