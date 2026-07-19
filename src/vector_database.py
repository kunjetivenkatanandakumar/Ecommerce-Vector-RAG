import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


def build_vector_database(input_path, db_path):
    print("Loading embedding dataset...")

    df = pd.read_pickle(input_path)

    print(f"Total records before removing duplicates: {len(df)}")

    # Remove duplicate product IDs
    df = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)

    print(f"Total records after removing duplicates: {len(df)}")

    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(path=db_path)

    # Delete old collection if it exists
    try:
        client.delete_collection("amazon_products")
        print("Old collection deleted.")
    except:
        print("No existing collection found.")

    # Create a new collection
    collection = client.create_collection("amazon_products")

    # Generate unique IDs for ChromaDB
    ids = [f"product_{i}" for i in range(len(df))]

    # Embeddings
    embeddings = df["embedding"].tolist()

    # Documents
    documents = df["combined_text"].tolist()

    # Metadata
    metadatas = []

    for _, row in df.iterrows():
        metadatas.append({
            "product_name": str(row["product_name"]),
            "category": str(row["category"]),
            "price": float(row["discounted_price"]),
            "rating": float(row["rating"])
        })

    print("Adding vectors to ChromaDB...")

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print("\n===================================")
    print("Vector Database Created Successfully!")
    print("===================================")
    print("Total Records:", collection.count())

    return collection


if __name__ == "__main__":

    input_file = r"C:\Users\kunja\Downloads\Ecommerce_Vector_RAG\data\processed\amazon_products_embeddings.pkl"

    db_path = r"C:\Users\kunja\Downloads\Ecommerce_Vector_RAG\chroma_db"

    collection = build_vector_database(input_file, db_path)

    print("\nLoading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    while True:
        query = input("\nEnter product search (or type 'exit'): ")

        if query.lower() == "exit":
            break

        query_embedding = model.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        print("\nTop 5 Similar Products\n")

        for i in range(len(results["ids"][0])):
            print("=" * 60)
            print("Product :", results["metadatas"][0][i]["product_name"])
            print("Category:", results["metadatas"][0][i]["category"])
            print("Price   :", results["metadatas"][0][i]["price"])
            print("Rating  :", results["metadatas"][0][i]["rating"])