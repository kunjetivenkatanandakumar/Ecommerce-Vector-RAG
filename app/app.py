import streamlit as st
import google.generativeai as genai
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.search import ProductSearchEngine
import streamlit as st

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Shopping Assistant",
    page_icon="🛒",
    layout="wide"
)

# -----------------------------
# Configure Gemini
# -----------------------------
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# -----------------------------
# Load Search Engine (cached)
# -----------------------------
@st.cache_resource
def load_search_engine():
    db_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "chroma_db"
    )
    return ProductSearchEngine(db_path)

search_engine = load_search_engine()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🛒 AI Shopping Assistant")

st.sidebar.markdown("### Technology Stack")
st.sidebar.write("✅ ChromaDB")
st.sidebar.write("✅ Sentence Transformers")
st.sidebar.write("✅ Google Gemini")
st.sidebar.write("✅ Streamlit")

st.sidebar.markdown("---")
st.sidebar.info(
    "Search products using natural language."
)

# -----------------------------
# Main Page
# -----------------------------
st.title("🛒 AI Shopping Assistant")

st.write(
    "Ask for any product in natural language and get AI-powered recommendations."
)

query = st.text_input(
    "Enter your product query",
    placeholder="Example: Suggest a wireless headphone under ₹2000"
)

# -----------------------------
# Search Button
# -----------------------------
if st.button("🔍 Search"):

    if query.strip() == "":
        st.warning("Please enter a search query.")
        st.stop()

    with st.spinner("Searching products..."):

        products = search_engine.search(query, top_k=5)

    if len(products) == 0:
        st.error("No matching products found.")
        st.stop()

    # -------------------------
    # Build Context
    # -------------------------
    context = ""

    for i, product in enumerate(products, start=1):

        context += f"""
Product {i}

Name: {product['product_name']}
Category: {product['category']}
Price: ₹{product['price']}
Rating: {product['rating']}
"""

    prompt = f"""
You are an AI Shopping Assistant.

Use ONLY the product information below.

{context}

Customer Question:
{query}

Instructions:
- Recommend the best products.
- Explain why they are suitable.
- Mention price and rating.
- Do not invent products.
- Keep the answer concise.
"""

    with st.spinner("Generating AI Recommendation..."):
        response = model.generate_content(prompt)

    # -------------------------
    # AI Recommendation
    # -------------------------
    st.subheader("🤖 AI Recommendation")

    st.success(response.text)

    st.divider()

    # -------------------------
    # Similar Products
    # -------------------------
    st.subheader("📦 Similar Products")

    for index, product in enumerate(products, start=1):

        with st.container():

            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"### {index}. {product['product_name']}")
                st.write(f"**Category:** {product['category']}")

            with col2:
                st.metric("Price", f"₹{product['price']:.2f}")
                st.metric("Rating", f"⭐ {product['rating']}")

            st.divider()

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Built using Streamlit, ChromaDB, Sentence Transformers, and Google Gemini."
)
