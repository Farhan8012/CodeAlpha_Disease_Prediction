import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the clean data
df = pd.read_csv("heart_cleaned.csv")

# 2. Separate Features (X) and Target (y)
X = df.drop('target', axis=1)  # All columns except 'target'
y = df['target']               # Only the 'target' column

# 3. Split into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale the data (Crucial for SVM and Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("--- Model Training Started ---")

# --- MODEL 1: Logistic Regression ---
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)
acc_log = accuracy_score(y_test, y_pred_log)

print(f"\n1. Logistic Regression Accuracy: {acc_log:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_log))

# --- MODEL 2: Support Vector Machine (SVM) ---
svm_model = SVC(kernel='linear') # Linear kernel often works well for simple classification
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, y_pred_svm)

print(f"\n2. SVM Accuracy: {acc_svm:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_svm))