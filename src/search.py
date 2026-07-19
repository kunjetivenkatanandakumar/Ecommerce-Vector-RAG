import os
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


class ProductSearchEngine:

    def __init__(self, db_path):

        self.db_path = db_path

        self.client = chromadb.PersistentClient(path=db_path)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        try:
            self.collection = self.client.get_collection("amazon_products")
            print("Existing ChromaDB loaded.")

        except Exception:

            print("Collection not found.")
            print("Creating ChromaDB...")

            self.collection = self.build_database()

    def build_database(self):

        data_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "processed",
            "amazon_products_embeddings.pkl"
        )

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

        print("Database created successfully.")

        return collection

    def search(self, query, top_k=5):

        embedding = self.model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )

        output = []

        for metadata in results["metadatas"][0]:

            output.append({
                "product_name": metadata["product_name"],
                "category": metadata["category"],
                "price": metadata["price"],
                "rating": metadata["rating"]
            })

        return output
