import pandas as pd
import numpy as np

np.random.seed(42)

n = 500

data = {
    "CGPA": np.round(np.random.uniform(5.5, 9.8, n), 2),
    "Attendance": np.random.randint(60, 101, n),
    "Aptitude_Score": np.random.randint(35, 101, n),
    "Technical_Skill_Score": np.random.randint(30, 101, n),
    "Communication_Score": np.random.randint(30, 101, n),
    "Projects": np.random.randint(0, 6, n),
    "Certifications": np.random.randint(0, 6, n),
    "Internships": np.random.randint(0, 3, n)
}

df = pd.DataFrame(data)

# Calculate placement probability
score = (
    df["CGPA"] * 7
    + df["Attendance"] * 0.5
    + df["Aptitude_Score"] * 0.7
    + df["Technical_Skill_Score"] * 0.8
    + df["Communication_Score"] * 0.5
    + df["Projects"] * 5
    + df["Certifications"] * 3
    + df["Internships"] * 6
)

probability = (score - score.min()) / (score.max() - score.min()) * 100

df["Placement_Probability"] = np.round(probability, 2)

# Placement status
df["Placement_Status"] = np.where(
    df["Placement_Probability"] >= 55,
    "Placed",
    "Not Placed"
)

df.to_csv("placement_data.csv", index=False)

print("Dataset created successfully!")
print(f"Total students: {len(df)}")
print("\nFirst 5 records:")
print(df.head())