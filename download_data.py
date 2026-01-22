from ucimlrepo import fetch_ucirepo
import pandas as pd

print("Fetching data from UCI Repository...")

# 1. Fetch dataset directly using the ID 45 (Heart Disease)
heart_disease = fetch_ucirepo(id=45)

# 2. Get the data (features and targets are separate by default)
X = heart_disease.data.features
y = heart_disease.data.targets

# 3. Combine them into one simple table
print("Combining data...")
df = pd.concat([X, y], axis=1)

# 4. Save to CSV so we have a local copy
df.to_csv("heart.csv", index=False)
print("Success! 'heart.csv' saved.")
print(f"Dataset shape: {df.shape}")