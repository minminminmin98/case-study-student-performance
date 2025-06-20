import pandas as pd
import os

def check_raw_data(filepath="data/student_dataset.csv"):
    """
    Loads the dataset and performs basic data checks without modifying it.

    Args:
        filepath (str): Path to the input dataset.

    Returns:
        pd.DataFrame: The original DataFrame.
    """
    # Load the dataset
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"\n🟢 Loaded dataset: {filepath}")
    print(f"Dataset shape: {df.shape}")

    # Basic data checks
    print("\n📋 Data Types:")
    print(df.dtypes)

    print("\n🧮 Missing Values:")
    print(df.isnull().sum())

    print("\n🔁 Duplicate Rows: ", df.duplicated().sum())

    return df

if __name__ == "__main__":
    check_raw_data()
