import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from feature_encoding import encode_student_data

def test_encoding_creates_output_file(tmp_path):
    # Prepare test input path and dummy data
    test_input_path = tmp_path / "student_dataset.csv"
    test_output_path = tmp_path / "processed_dataset.csv"

    # Create a minimal sample dataset with expected columns
    sample_data = pd.DataFrame({
        'gender': ['M', 'F'],
        'NationalITy': ['Kuwait', 'USA'],
        'PlaceofBirth': ['Kuwait', 'USA'],
        'StageID': ['HighSchool', 'MiddleSchool'],
        'GradeID': ['G-10', 'G-08'],
        'SectionID': ['A', 'B'],
        'Topic': ['Math', 'Science'],
        'Semester': ['F', 'S'],
        'Relation': ['Father', 'Mum'],
        'ParentAnsweringSurvey': ['Yes', 'No'],
        'ParentschoolSatisfaction': ['Good', 'Bad'],
        'StudentAbsenceDays': ['Under-7', 'Above-7']
    })
    sample_data.to_csv(test_input_path, index=False)

    # Run the encoding function
    df_encoded = encode_student_data(str(test_input_path), str(test_output_path))

    # Assertions
    assert os.path.exists(test_output_path), "Output file was not created"
    assert df_encoded.isnull().sum().sum() == 0, "There are missing values in the encoded dataset"
    for col in df_encoded.columns:
        assert pd.api.types.is_numeric_dtype(df_encoded[col]), f"Non-numeric data in column '{col}'"

    print("✅ test_encoding_creates_output_file passed.")
