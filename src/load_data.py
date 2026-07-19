import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    df = load_dataset("C:/Users/kunja/Downloads/Ecommerce_Vector_RAG/data/raw/amazon_products.csv")
    print(df.head())
    print(df.shape)