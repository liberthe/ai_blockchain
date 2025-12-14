import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report

# Load data
print("Loading data...")
df = pd.read_csv('final_thesis_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# Prepare features and target
features = ['Age', 'Credit amount', 'Duration', 'Telco_Bill', 'Social_Score']
X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
y = df['Target']

print(f"Target distribution:\n{y.value_counts()}\n")

# Split data
print("Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
print("Training RandomForest model (100 trees)...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
recall_0 = recall_score(y_test, y_pred, pos_label=0)  # recall for class 0 (bad)
recall_1 = recall_score(y_test, y_pred, pos_label=1)  # recall for class 1 (good)
recall_macro = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Recall (Macro): {recall_macro:.4f}")
print(f"  - Recall for Class 0 (Bad): {recall_0:.4f}")
print(f"  - Recall for Class 1 (Good): {recall_1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")
print("="*60 + "\n")

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print()

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Bad (0)', 'Good (1)']))

print("\nFeature Importances:")
for feat, importance in zip(features, model.feature_importances_):
    print(f"  {feat}: {importance:.4f}")

print("\nSample predictions (first 10 test samples):")
print(f"{'Actual':<10} {'Predicted':<12} {'Probability':<15}")
print("-" * 40)
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    prob = y_pred_proba[i]
    print(f"{actual:<10} {predicted:<12} {prob:.4f}")
