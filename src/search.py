import chromadb
from sentence_transformers import SentenceTransformer


class ProductSearchEngine:

    def __init__(self, db_path):

        print("Loading ChromaDB...")

        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_collection(
            name="amazon_products"
        )

        print("Loading Embedding Model...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Search Engine Ready!\n")

    def search(self, query, top_k=5):

        query_embedding = self.model.encode(
            query
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        output = []

        for i in range(len(results["ids"][0])):

            metadata = results["metadatas"][0][i]

            output.append({

                "product_name": metadata["product_name"],

                "category": metadata["category"],

                "price": metadata["price"],

                "rating": metadata["rating"]

            })

        return output


if __name__ == "__main__":

    search_engine = ProductSearchEngine(
        r"C:\Users\kunja\Downloads\Ecommerce_Vector_RAG\chroma_db"
    )

    while True:

        query = input("\nSearch Product (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        products = search_engine.search(query)

        print("\nTop Recommendations\n")

        for i, product in enumerate(products, start=1):

            print("=" * 60)

            print(f"{i}. {product['product_name']}")

            print("Category :", product["category"])

            print("Price    :", product["price"])

            print("Rating   :", product["rating"])