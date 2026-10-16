
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 1. Load the dataset
train = pd.read_csv("data/train.csv")

print("Training data:", train.shape)
print(train.head())

print("\nSurvival counts:")
print(train["Survived"].value_counts())

# 2. Select features and target
features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = train[features]
y = train["Survived"]

print("\nFeatures:", X.shape)
print("Target:", y.shape)

print("\nMissing values:")
print(X.isnull().sum())

# 3. Define numeric and categorical features
numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Sex",
    "Embarked"
]

# 4. Preprocess numeric features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# 5. Preprocess categorical features
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 6. Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 7. Create the machine learning model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# 8. Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 9. Train the model
model.fit(X_train, y_train)

# 10. Make predictions
predictions = model.predict(X_valid)

# 11. Evaluate accuracy
accuracy = accuracy_score(y_valid, predictions)

print("\nValidation Accuracy:", accuracy)

# 12. Evaluate model performance
print("\nConfusion Matrix:")
print(confusion_matrix(y_valid, predictions))

print("\nClassification Report:")
print(classification_report(y_valid, predictions))







import numpy as np
import matplotlib.pyplot as plt

# Survival rate by passenger class
survival_by_class = train.groupby("Pclass")["Survived"].mean()

# Survival rate by gender
survival_by_sex = train.groupby("Sex")["Survived"].mean()

print("Survival rate by class:")
print(survival_by_class)

print("\nSurvival rate by gender:")
print(survival_by_sex)

# Numerical analysis with NumPy
survival_rates = survival_by_class.to_numpy()

print("\nAverage survival rate across classes:",
      np.mean(survival_rates))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

survival_by_class.plot(
    kind="bar",
    ax=axes[0],
    title="Survival Rate by Passenger Class"
)

survival_by_sex.plot(
    kind="bar",
    ax=axes[1],
    title="Survival Rate by Gender"
)

axes[0].set_ylabel("Survival Rate")
axes[1].set_ylabel("Survival Rate")

plt.tight_layout()
plt.show()