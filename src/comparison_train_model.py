# comparison_train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data/processed_dataset.csv")  # Your actual dataset path

# Features and target
X = df.drop(['Class', 'ID'], axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Classifiers to compare
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True, random_state=42)
}

results_summary = []

# Train and evaluate
for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    # Collect results
    results_summary.append(f"\n--- {name} ---\nAccuracy: {accuracy:.4f}\n")
    results_summary.append("Classification Report:\n" + report)
    results_summary.append("Confusion Matrix:\n" + str(cm) + "\n")

# Save evaluation results
with open("model_comparison_report.txt", "w") as f:
    f.writelines(results_summary)

print("\n✅ Model comparison complete. Results saved to model_comparison_report.txt.")
