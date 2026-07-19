import pandas as pd
import re


def clean_text(text):
    """
    Clean text by:
    - Converting to lowercase
    - Removing HTML tags
    - Removing special characters
    - Removing extra spaces
    """
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_data(input_path, output_path):
    print("Loading dataset...")
    df = pd.read_csv(r"C:/Users/kunja/Downloads/Ecommerce_Vector_RAG/data/raw/amazon_products.csv")

    print(f"Original Shape: {df.shape}")

    # Remove unwanted columns (only if they exist)
    columns_to_drop = [
        "user_id",
        "user_name",
        "review_id",
        "img_link",
        "product_link"
    ]

    df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    # Remove missing values
    df.dropna(inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # -----------------------------
    # Clean Price Columns
    # -----------------------------
    price_columns = ["actual_price", "discounted_price"]

    for col in price_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("₹", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -----------------------------
    # Clean Rating
    # -----------------------------
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # -----------------------------
    # Clean Rating Count
    # -----------------------------
    if "rating_count" in df.columns:
        df["rating_count"] = (
            df["rating_count"]
            .astype(str)
            .str.replace(",", "", regex=False)
        )
        df["rating_count"] = pd.to_numeric(
            df["rating_count"],
            errors="coerce"
        )

    # Remove rows created by numeric conversion errors
    df.dropna(inplace=True)

    # -----------------------------
    # Clean Text Columns
    # -----------------------------
    text_columns = [
        "product_name",
        "category",
        "about_product",
        "review_title",
        "review_content"
    ]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # -----------------------------
    # Create Combined Text
    # -----------------------------
    text_cols = [col for col in text_columns if col in df.columns]

    df["combined_text"] = df[text_cols].fillna("").agg(" ".join, axis=1)

    # Remove extra spaces from combined text
    df["combined_text"] = df["combined_text"].str.replace(
        r"\s+",
        " ",
        regex=True
    ).str.strip()

    # Save cleaned dataset
    df.to_csv(output_path, index=False)

    print("===================================")
    print("Preprocessing Completed Successfully!")
    print("===================================")
    print(f"Final Shape : {df.shape}")
    print(f"Saved File  : {output_path}")
    print("\nData Types:\n")
    print(df.dtypes)


if __name__ == "__main__":
    input_file = "C:/Users/kunja/Downloads/Ecommerce_Vector_RAG/data/raw/amazon_products.csv"
    output_file = "C:/Users/kunja/Downloads/Ecommerce_Vector_RAG/data/processed/amazon_products_clean.csv"

    preprocess_data(input_file, output_file)