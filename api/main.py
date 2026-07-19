from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import sys
import os

# Project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.search import ProductSearchEngine
from app.config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

search_engine = ProductSearchEngine(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "chroma_db"
    )
)

app = FastAPI(
    title="AI Shopping Assistant API",
    version="1.0"
)

class SearchRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {
        "message": "AI Shopping Assistant API is running!"
    }


@app.post("/recommend")
def recommend(request: SearchRequest):

    products = search_engine.search(
        request.query,
        top_k=5
    )

    context = ""

    for i, p in enumerate(products, start=1):

        context += f"""
Product {i}

Name: {p['product_name']}
Category: {p['category']}
Price: ₹{p['price']}
Rating: {p['rating']}
"""

    prompt = f"""
You are an AI Shopping Assistant.

Use ONLY these products.

{context}

Customer Question:

{request.query}

Recommend the best products.
Explain why.
Mention price and rating.
"""

    response = model.generate_content(prompt)

    return {
        "recommendation": response.text,
        "products": products
    }