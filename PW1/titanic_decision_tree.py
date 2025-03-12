import pandas as pd
import numpy as np
from math import log2

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Select relevant columns and handle missing values
df = df[["Survived", "Pclass", "Sex", "Age"]].dropna()
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Function to calculate entropy
def entropy(column):
    values, counts = np.unique(column, return_counts=True)
    probs = counts / counts.sum()
    return -sum(p * log2(p) for p in probs if p > 0)

# Function to calculate information gain
def information_gain(data, split_attr, target_attr="Survived"):
    total_entropy = entropy(data[target_attr])
    values, counts = np.unique(data[split_attr], return_counts=True)
    weighted_entropy = sum((counts[i] / sum(counts)) * entropy(data[data[split_attr] == values[i]][target_attr]) for i in range(len(values)))
    return total_entropy - weighted_entropy

# Compute entropy of Survived column
survival_entropy = entropy(df["Survived"])
print(f"Entropy of Survival: {survival_entropy}")

# Compute information gain for each attribute
attributes = ["Sex", "Pclass", "Age"]
info_gains = {attr: information_gain(df, attr) for attr in attributes}

# Find the best attribute for the first split
best_attr = max(info_gains, key=info_gains.get)
print("\nInformation Gain for each attribute:")
for attr, gain in info_gains.items():
    print(f"{attr}: {gain}")

print(f"\nBest attribute to split first: {best_attr}")
