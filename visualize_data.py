import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the CLEAN data
df = pd.read_csv("heart_cleaned.csv")

# 2. Create a Correlation Heatmap
# This shows how much each variable relates to the others
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt=".1f", cmap="coolwarm")
plt.title("Correlation Matrix (What relates to Heart Disease?)")

# Save it as an image so you can show it in your video later
plt.savefig("correlation_heatmap.png")
print("1. Correlation Heatmap saved as 'correlation_heatmap.png'")

# 3. Create a Target Balance Chart
# Checking if we have balanced classes
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, palette="pastel")
plt.title("Count of Patients (0 = Healthy, 1 = Disease)")

plt.savefig("target_distribution.png")
print("2. Target Distribution saved as 'target_distribution.png'")