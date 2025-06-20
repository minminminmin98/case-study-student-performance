import pandas as pd
import os

def encode_student_data(input_path="data/student_dataset.csv", output_path="data/processed_dataset.csv"):
    """
    Loads student_dataset.csv, applies label encoding to categorical features,
    checks for missing values after encoding, and saves the processed dataset.

    Args:
        input_path (str): Path to the raw dataset.
        output_path (str): Path to save the processed dataset.

    Returns:
        pd.DataFrame: Encoded dataset.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"🔍 Loaded dataset from {input_path}. Shape: {df.shape}")

    # Mappings
    mappings = {
        'gender': {'M': 0, 'F': 1},
        'NationalITy': {
            'Kuwait': 0, 'Lebanon': 1, 'Egypt': 2, 'Saudi Arabia': 3, 'USA': 4, 'Jordan': 5,
            'Venzuela': 6, 'Iran': 7, 'Tunis': 8, 'Morocco': 9, 'Syria': 10,
            'Palestine': 11, 'Iraq': 12, 'Lybia': 13
        },
        'PlaceofBirth': {
            'Kuwait': 0, 'Lebanon': 1, 'Egypt': 2, 'Saudi Arabia': 3, 'USA': 4, 'Jordan': 5,
            'Venzuela': 6, 'Iran': 7, 'Tunis': 8, 'Morocco': 9, 'Syria': 10,
            'Palestine': 11, 'Iraq': 12, 'Lybia': 13
        },
        'StageID': {'lowerlevel': 0, 'MiddleSchool': 1, 'HighSchool': 2},
        'GradeID': {
            'G-01': 1, 'G-02': 2, 'G-03': 3, 'G-04': 4, 'G-05': 5, 'G-06': 6,
            'G-07': 7, 'G-08': 8, 'G-09': 9, 'G-10': 10, 'G-11': 11, 'G-12': 12
        },
        'SectionID': {'A': 0, 'B': 1, 'C': 2},
        'Topic': {
            'English': 0, 'Spanish': 1, 'French': 2, 'Arabic': 3, 'IT': 4, 'Math': 5,
            'Chemistry': 6, 'Biology': 7, 'Science': 8, 'History': 9, 'Quran': 10, 'Geology': 11
        },
        'Semester': {'F': 0, 'S': 1},
        'Relation': {'Father': 0, 'Mum': 1},
        'ParentAnsweringSurvey': {'No': 0, 'Yes': 1},
        'ParentschoolSatisfaction': {'Bad': 0, 'Good': 1},
        'StudentAbsenceDays': {'Under-7': 0, 'Above-7': 1}
    }

    # Apply mappings
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
        else:
            print(f"⚠️ Column '{col}' not found in dataset. Skipping.")

    # Check for missing values after mapping
    missing = df.isnull().sum()
    total_missing = missing.sum()

    if total_missing > 0:
        print("⚠️ Missing values found after encoding:")
        print(missing[missing > 0])
    else:
        print("✅ No missing values after encoding.")

    print(f"✅ Encoding complete. Final shape: {df.shape}")

    # Save processed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Encoded dataset saved to: {output_path}")

    return df

# Run if executed directly
if __name__ == "__main__":
    encode_student_data()
