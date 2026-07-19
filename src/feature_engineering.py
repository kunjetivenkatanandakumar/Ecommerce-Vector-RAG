import pandas as pd


def feature_engineering(input_path, output_path):
    df = pd.read_csv(r"C:\\Users\\kunja\\Downloads\\Ecommerce_Vector_RAG\\data\\processed\\amazon_products_clean.csv")

    df["discount_percentage"] = (
        df["discount_percentage"]
        .astype(str)
        .str.replace("%", "", regex=False)
    )

    df["discount_percentage"] = pd.to_numeric(
        df["discount_percentage"],
        errors="coerce"
    )

    df["discount_amount"] = (
        df["actual_price"]
        - df["discounted_price"]
    )

    df["description_length"] = (
        df["about_product"]
        .fillna("")
        .str.len()
    )

    df["review_length"] = (
        df["review_content"]
        .fillna("")
        .str.len()
    )

    df["product_name_length"] = (
        df["product_name"]
        .fillna("")
        .str.len()
    )

    df["category_depth"] = (
        df["category"]
        .astype(str)
        .str.count(r"\|")
        + 1
    )

    df["combined_text"] = (
        "Product Name: " + df["product_name"] +
        ". Category: " + df["category"] +
        ". Description: " + df["about_product"] +
        ". Review Title: " + df["review_title"] +
        ". Review: " + df["review_content"] +
        ". Rating: " + df["rating"].astype(str)
    )

    df["combined_text"] = (
        df["combined_text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df.to_csv(output_path, index=False)

    print("Feature Engineering Completed!")
    print("Final Shape:", df.shape)


if __name__ == "__main__":
    feature_engineering(
        "C:\\Users\\kunja\\Downloads\\Ecommerce_Vector_RAG\\data\\processed\\amazon_products_clean.csv",
        "C:\\Users\\kunja\\Downloads\\Ecommerce_Vector_RAG\\data\\processed\\amazon_products_features.csv"
    )