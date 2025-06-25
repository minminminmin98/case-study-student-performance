import os
import sys
import joblib

# Add src directory to path for import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_best_model import train_and_evaluate

def test_train_and_evaluate_creates_model_and_returns_accuracy():
    # Remove model file if it exists from a previous run
    model_path = "models/best_random_forest_model.pkl"
    if os.path.exists(model_path):
        os.remove(model_path)

    # Run training
    accuracy = train_and_evaluate()

    # Assertions
    assert os.path.exists(model_path), "Trained model file was not created."
    assert isinstance(accuracy, float), "Accuracy should be a float value."
    assert 0.0 <= accuracy <= 1.0, f"Accuracy {accuracy} out of expected range."

    # Optional: load the model to confirm it can be unpickled
    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "Loaded model does not have a predict method."

    print("✅ test_train_and_evaluate_creates_model_and_returns_accuracy passed.")
