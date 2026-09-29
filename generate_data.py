import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "IQ":                   np.random.normal(105, 12, n).round().clip(80, 140).astype(int),
    "CGPA":                 np.random.normal(7.2, 1.0, n).round(2).clip(4.0, 10.0),
    "10th_Marks":           np.random.normal(78, 10, n).round(1).clip(45, 99),
    "12th_Marks":           np.random.normal(75, 11, n).round(1).clip(45, 99),
    "Communication_Skills": np.random.randint(1, 11, n),   # rating 1-10
})

# Hidden rule: a weighted score + noise decides placement
score = (
    0.45 * (df["CGPA"] - 4) / 6
    + 0.25 * (df["Communication_Skills"] - 1) / 9
    + 0.15 * (df["IQ"] - 80) / 60
    + 0.10 * (df["12th_Marks"] - 45) / 54
    + 0.05 * (df["10th_Marks"] - 45) / 54
    + np.random.normal(0, 0.05, n)          # noise
)
df["Placed"] = (score > 0.50).astype(int)   # 1 = placed, 0 = not placed

df.to_csv("data/placement.csv", index=False)
print(df.head())
print("\nPlaced ratio:", df["Placed"].mean().round(2))