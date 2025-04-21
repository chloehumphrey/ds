import pandas as pd
from preprocess import preprocess_data

# Import the dataset into a pandas DataFrame
df = pd.read_csv('./home-credit-default-risk/application_train.csv')

# Display the first few rows of the DataFrame
print("First few rows of the dataset...")
print(df.head())
print("\nDataFrame info...")
print(df.shape)

def load_data(path):
    # Load the dataset into a pandas DataFrame
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    
    path = './home-credit-default-risk/application_train.csv'
    df = load_data(path)
    # Preprocess the dataset
    df = preprocess_data(df)

    for col in df.columns:
        print(f"{col}: {df[col].dtype}")