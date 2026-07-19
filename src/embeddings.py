import pandas as pd
from sentence_transformers import SentenceTransformer


def generate_embeddings(input_path, output_path):

    print("Loading dataset...")

    df = pd.read_csv(input_path)

    print("Loading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")

    embeddings = model.encode(
        df["combined_text"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True
    )

    df["embedding"] = embeddings.tolist()

    df.to_pickle(output_path)

    print("\nEmbedding Generation Completed!")
    print("Embedding Shape:", embeddings.shape)
    print("Saved to:", output_path)


if __name__ == "__main__":

    input_file = "C:\\Users\\kunja\\Downloads\\Ecommerce_Vector_RAG\\data\\processed\\amazon_products_features.csv"

    output_file = "C:\\Users\\kunja\\Downloads\\Ecommerce_Vector_RAG\\data\\processed\\amazon_products_embeddings.pkl"

    generate_embeddings(input_file, output_file)