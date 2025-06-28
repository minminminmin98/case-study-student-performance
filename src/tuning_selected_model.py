import pandas as pd
import os
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ✅ Set working directory info (for debug if needed)
print("📁 Working directory:", os.getcwd())

# ✅ Load dataset
df_processed = pd.read_csv("data/processed_dataset.csv")

# === Define the features to include (behavioral only) ===
features_to_include = ['VisITedResources', 'raisedhands', 'AnnouncementsView', 'StudentAbsenceDays', 'Discussion']

# === Separate features and target ===
X = df_processed[features_to_include]
y = df_processed['Class']

# ✅ Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Define hyperparameter grid
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# ✅ Create GridSearchCV
grid_search_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_rf,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

# ✅ Run GridSearch
print("🔍 Tuning Random Forest...")
grid_search_rf.fit(X_train_scaled, y_train)

# ✅ Get best model and evaluate
best_rf_model = grid_search_rf.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test_scaled)

# ✅ Print results
print("\n✅ Best Hyperparameters:")
print(grid_search_rf.best_params_)

print("\n📊 Evaluation on Test Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best_rf))

with open("rf_tuning_results.txt", "w", encoding="utf-8") as f:
    f.write("✅ Best Hyperparameters:\n")
    f.write(str(grid_search_rf.best_params_) + "\n\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_best_rf):.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred_best_rf) + "\n")
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_best_rf)))
