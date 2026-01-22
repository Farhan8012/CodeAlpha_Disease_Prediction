import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Load Data
df = pd.read_csv("heart_cleaned.csv")
X = df.drop('target', axis=1)
y = df['target']

# 2. Split (Same as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale (CRITICAL for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. Train the Winner (SVM)
print("Training the best model (SVM)...")
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)

# 5. Save Model and Scaler to files
pickle.dump(svm_model, open('svm_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Success! Model saved as 'svm_model.pkl' and Scaler saved as 'scaler.pkl'")