import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_and_evaluate():
    # Ensure model folder exists
    os.makedirs("models", exist_ok=True)

    # Load the processed dataset
    df = pd.read_csv("data/processed_dataset.csv")

    # === Define the features to include (behavioral only) ===
    features_to_include = ['VisITedResources', 'raisedhands', 'AnnouncementsView', 'StudentAbsenceDays', 'Discussion']

    # === Separate features and target ===
    X = df[features_to_include]
    y = df['Class']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Best hyperparameters from tuning
    best_rf = RandomForestClassifier(
        bootstrap=True,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=300,
        random_state=42
    )

    # Train the model
    print("🚀 Training best Random Forest model...")
    best_rf.fit(X_train, y_train)

    # Save the model
    model_path = "models/best_random_forest_model.pkl"
    joblib.dump(best_rf, model_path)
    print(f"✅ Model saved to {model_path}")

    # Evaluate the model
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print results
    print("\n📊 Evaluation on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Save results to a text file
    with open("best_model_results.txt", "w", encoding="utf-8") as f:
        f.write("📊 Evaluation on Test Set:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix))

    return accuracy  # Return for test assertion

# Optional: run directly as script
if __name__ == "__main__":
    train_and_evaluate()
