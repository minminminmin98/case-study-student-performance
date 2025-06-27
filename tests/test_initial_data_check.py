import pandas as pd

def test_dataset_not_empty():
    df = pd.read_csv("data/processed_dataset.csv")
    assert not df.empty, "Dataset is empty"
    print("✅ Data is loaded correctly and not empty.")

if __name__ == "__main__":
    test_dataset_not_empty()