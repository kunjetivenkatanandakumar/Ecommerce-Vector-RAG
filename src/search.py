import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


class ProductSearchEngine:

    def __init__(self, db_path):

        self.client = chromadb.PersistentClient(path=db_path)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        try:
            self.collection = self.client.get_collection("amazon_products")
            print("Loaded existing ChromaDB collection.")

        except Exception:
            print("Collection not found. Creating a new one...")
            self.collection = self.build_database()

    def build_database(self):

        data_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "processed",
                "amazon_products_embeddings.pkl"
            )
        )

        print("Loading:", data_path)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Embedding file not found: {data_path}")

        df = pd.read_pickle(data_path)
        df = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)

        collection = self.client.create_collection("amazon_products")

        collection.add(
            ids=[f"product_{i}" for i in range(len(df))],
            embeddings=df["embedding"].tolist(),
            documents=df["combined_text"].tolist(),
            metadatas=[
                {
                    "product_name": str(row["product_name"]),
                    "category": str(row["category"]),
                    "price": float(row["discounted_price"]),
                    "rating": float(row["rating"])
                }
                for _, row in df.iterrows()
            ]
        )

        print("Collection created successfully.")
        return collection

    def search(self, query, top_k=5):

        query_embedding = self.model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results["metadatas"][0]
