import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === Load preprocessed dataset ===
data_path = "data/processed_dataset.csv"
df = pd.read_csv(data_path)

# === Separate features and target ===
X = df.drop(['Class', 'ID'], axis=1)
y = df['Class']

# === Set test size ===
test_size = 0.2

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# === Feature scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Define classifiers ===
classifiers_to_run = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine (SVC)": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# === Train and evaluate ===
results_subset = {}
for name, clf in classifiers_to_run.items():
    print(f"\n🔄 Training {name} with test_size={test_size}...")
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    results_subset[name] = {
        "Accuracy": accuracy,
        "Classification Report": report,
        "Confusion Matrix": confusion
    }

    print(f"✅ Accuracy: {accuracy:.4f}")
    print("📋 Classification Report:")
    print(report)
    print("📊 Confusion Matrix:")
    print(confusion)

# === Summary table ===
print("\n--- 📊 Selected Classifier Accuracy Summary ---")
table_data = [['Classifier', f'Accuracy (test_size={test_size})']]
for name, result in results_subset.items():
    table_data.append([name, f"{result['Accuracy']:.4f}"])

df_results = pd.DataFrame(table_data[1:], columns=table_data[0])
print(df_results.to_markdown(index=False))

# Save evaluation results
report_path = "model_comparison_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Model Comparison Report (test_size={test_size})\n")
    f.write("=" * 50 + "\n\n")

    for name, result in results_subset.items():
        f.write(f"{name}\n")
        f.write(f"Accuracy: {result['Accuracy']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(result["Classification Report"] + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(result["Confusion Matrix"]) + "\n")
        f.write("\n" + "-" * 50 + "\n\n")

    # Plain summary table
    f.write("Accuracy Summary Table:\n")
    f.write("{:<35} {:>10}\n".format("Classifier", f"Accuracy"))
    f.write("-" * 50 + "\n")
    for clf_name, result in results_subset.items():
        f.write("{:<35} {:>10.4f}\n".format(clf_name, result["Accuracy"]))

print(f"✅ Model comparison report saved to {report_path}")