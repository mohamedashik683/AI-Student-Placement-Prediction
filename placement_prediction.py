import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
df = pd.read_csv("placement_data.csv")

# Features and target
features = [
    "CGPA",
    "Attendance",
    "Aptitude_Score",
    "Technical_Skill_Score",
    "Communication_Score",
    "Projects",
    "Certifications",
    "Internships"
]

X = df[features]
y = df["Placement_Status"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)

logistic_pred = logistic_model.predict(X_test_scaled)

# Random Forest
random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

random_forest_model.fit(X_train, y_train)

random_forest_pred = random_forest_model.predict(X_test)

# Evaluation function
def evaluate_model(name, y_test, predictions):
    print("\n" + "=" * 40)
    print(name)
    print("=" * 40)

    print("Accuracy :", round(
        accuracy_score(y_test, predictions) * 100, 2
    ), "%")

    print("Precision:", round(
        precision_score(y_test, predictions, pos_label="Placed") * 100, 2
    ), "%")

    print("Recall   :", round(
        recall_score(y_test, predictions, pos_label="Placed") * 100, 2
    ), "%")

    print("F1 Score :", round(
        f1_score(y_test, predictions, pos_label="Placed") * 100, 2
    ), "%")


# Compare models
evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_pred
)

evaluate_model(
    "Random Forest",
    y_test,
    random_forest_pred
)

print("\nModel training and evaluation completed successfully!")