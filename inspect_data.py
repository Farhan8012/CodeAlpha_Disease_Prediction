import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv("heart.csv")

print("--- Data Loaded ---")
print(f"Initial shape: {df.shape}")

# 2. FIX: Force all columns to be numeric
# If a value is '?' or weird text, turn it into NaN (Not a Number) so we can handle it.
# We apply this to the whole dataframe to be safe.
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\n--- Checking for Missing Values (After Numeric Conversion) ---")
print(df.isnull().sum())

# 3. Create Target Column (if 'num' exists)
# In this dataset, 'num' > 0 means heart disease.
if 'num' in df.columns:
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop(columns=['num'])
    print("\nTarget column created. 1 = Disease, 0 = Healthy")

# 4. Fill Missing Values
# Now that everything is a number, we can safely calculate the mean.
df = df.fillna(df.mean())

# 5. Save the clean data
df.to_csv("heart_cleaned.csv", index=False)
print("\nSuccess! Cleaned data saved as 'heart_cleaned.csv'.")