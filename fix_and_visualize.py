import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv("heart_cleaned.csv")

# 2. FIX: Check if 'target' exists. If not, create it from the last column.
if 'target' not in df.columns:
    print("'target' column missing. Fixing it now...")
    # The last column in this dataset is always the disease label
    last_col = df.columns[-1]
    print(f"Using column '{last_col}' as target.")
    
    # Rename it to 'target'
    df = df.rename(columns={last_col: 'target'})
    
    # Normalize: 0 = Healthy, 1+ = Disease (Binary Classification)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # Save the fixed file so your future models work!
    df.to_csv("heart_cleaned.csv", index=False)
    print("Fixed data saved to 'heart_cleaned.csv'.")
else:
    print("'target' column already exists. Proceeding...")

# 3. Create Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt=".1f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("correlation_heatmap.png")
print("1. Correlation Heatmap saved as 'correlation_heatmap.png'")

# 4. Create Target Balance Chart
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, palette="pastel")
plt.title("Count of Patients (0 = Healthy, 1 = Disease)")
plt.savefig("target_distribution.png")
print("2. Target Distribution saved as 'target_distribution.png'")