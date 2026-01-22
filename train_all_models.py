import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the clean data
df = pd.read_csv("heart_cleaned.csv")

# 2. Prepare Data
X = df.drop('target', axis=1)
y = df['target']

# 3. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale Data (Important for SVM & Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("--- Training All Models ---")

# --- MODEL 1: Logistic Regression ---
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
log_acc = accuracy_score(y_test, log_model.predict(X_test_scaled))
print(f"1. Logistic Regression: {log_acc:.2f}")

# --- MODEL 2: SVM ---
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)
svm_acc = accuracy_score(y_test, svm_model.predict(X_test_scaled))
print(f"2. SVM:                 {svm_acc:.2f}")

# --- MODEL 3: Random Forest (No scaling needed usually, but fine to use) ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # Note: RF doesn't strictly need scaled data
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"3. Random Forest:       {rf_acc:.2f}")

# --- MODEL 4: XGBoost ---
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
print(f"4. XGBoost:             {xgb_acc:.2f}")

print("\n--- Best Model? ---")
models = {'Logistic Regression': log_acc, 'SVM': svm_acc, 'Random Forest': rf_acc, 'XGBoost': xgb_acc}
best_model_name = max(models, key=models.get)
print(f"The winner is: {best_model_name} with accuracy {models[best_model_name]:.2f}")