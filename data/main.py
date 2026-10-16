
# ============================================================
# TITANIC SURVIVAL PREDICTION
# Libraries: Pandas, NumPy, Matplotlib, Scikit-learn, PyTorch
# ============================================================

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# -------------------- 1. Configuration --------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("TITANIC SURVIVAL PREDICTION")
print("=" * 60)
print("PyTorch device:", DEVICE)

# -------------------- 2. Load Data --------------------

train = pd.read_csv("data/train.csv")

print("\nDataset shape:", train.shape)
print("\nFirst five rows:")
print(train.head())

print("\nDataset information:")
train.info()

print("\nMissing values:")
print(train.isnull().sum())

print("\nTarget distribution:")
print(train["Survived"].value_counts())

# -------------------- 3. Exploratory Data Analysis --------------------

survival_by_class = train.groupby("Pclass")["Survived"].mean()
survival_by_sex = train.groupby("Sex")["Survived"].mean()

print("\nSurvival rate by passenger class:")
print(survival_by_class)

print("\nSurvival rate by sex:")
print(survival_by_sex)

# NumPy: numerical summary
survival_rates = survival_by_class.to_numpy()

print("\nMean survival rate across passenger classes:",
      np.mean(survival_rates))
print("Standard deviation:",
      np.std(survival_rates))

# Plot survival patterns
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

survival_by_class.plot(
    kind="bar",
    ax=axes[0],
    title="Survival Rate by Class",
)
axes[0].set_ylabel("Survival Rate")
axes[0].set_ylim(0, 1)

survival_by_sex.plot(
    kind="bar",
    ax=axes[1],
    title="Survival Rate by Sex",
)
axes[1].set_ylabel("Survival Rate")
axes[1].set_ylim(0, 1)

train["Age"].plot(
    kind="hist",
    bins=25,
    ax=axes[2],
    title="Passenger Age Distribution",
)
axes[2].set_xlabel("Age")

plt.tight_layout()
plt.show()

# -------------------- 4. Feature Engineering --------------------

def create_features(data):
    data = data.copy()

    # Family size
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

    # Whether the passenger travelled alone
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    # Extract passenger title from name
    data["Title"] = data["Name"].str.extract(r",\s*([^.]*)\.")

    # Group uncommon titles
    common_titles = ["Mr", "Miss", "Mrs", "Master"]
    data["Title"] = data["Title"].replace(
        {title: title for title in common_titles}
    )
    data.loc[~data["Title"].isin(common_titles), "Title"] = "Other"

    # Extract cabin deck; missing cabins get their own category
    data["Deck"] = data["Cabin"].str[0].fillna("Unknown")

    return data


train_features = create_features(train)

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "FamilySize",
    "IsAlone",
    "Title",
    "Deck",
]

X = train_features[features]
y = train_features["Survived"].astype("int64")

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
]

categorical_features = [
    "Sex",
    "Embarked",
    "Title",
    "Deck",
]

# Split before fitting preprocessing to avoid data leakage
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y,
)

# -------------------- 5. Scikit-learn Model --------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            random_state=SEED,
            class_weight="balanced",
        ),
    ),
])

print("\nTraining Random Forest...")
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_valid)
rf_accuracy = accuracy_score(y_valid, rf_predictions)

print("\nRandom Forest Accuracy:", round(rf_accuracy, 4))
print("\nRandom Forest Classification Report:")
print(classification_report(y_valid, rf_predictions))

print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_valid, rf_predictions))

# -------------------- 6. Prepare Data for PyTorch --------------------

# Fit preprocessing using training data only
torch_preprocessor = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]),
        numeric_features,
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]),
        categorical_features,
    ),
])

X_train_np = torch_preprocessor.fit_transform(X_train)
X_valid_np = torch_preprocessor.transform(X_valid)

# Convert sparse matrices if necessary
if hasattr(X_train_np, "toarray"):
    X_train_np = X_train_np.toarray()

if hasattr(X_valid_np, "toarray"):
    X_valid_np = X_valid_np.toarray()

# NumPy: ensure correct dtype and shape
X_train_np = np.asarray(X_train_np, dtype=np.float32)
X_valid_np = np.asarray(X_valid_np, dtype=np.float32)

y_train_np = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
y_valid_np = y_valid.to_numpy(dtype=np.float32).reshape(-1, 1)

X_train_tensor = torch.from_numpy(X_train_np)
y_train_tensor = torch.from_numpy(y_train_np)

X_valid_tensor = torch.from_numpy(X_valid_np).to(DEVICE)

# -------------------- 7. Build Neural Network --------------------

class TitanicNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


input_size = X_train_tensor.shape[1]

nn_model = TitanicNet(input_size).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    nn_model.parameters(),
    lr=0.001,
    weight_decay=1e-4,
)

# Use a fixed validation set throughout training
train_dataset = torch.utils.data.TensorDataset(
    X_train_tensor,
    y_train_tensor,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)

# -------------------- 8. Train PyTorch Model --------------------

EPOCHS = 100
best_accuracy = -1
best_state = None
patience = 15
epochs_without_improvement = 0

history = {
    "loss": [],
    "validation_accuracy": [],
}

print("\nTraining PyTorch Neural Network...")

for epoch in range(EPOCHS):
    nn_model.train()
    total_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        logits = nn_model(batch_X)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_X)

    average_loss = total_loss / len(train_dataset)

    nn_model.eval()

    with torch.no_grad():
        validation_logits = nn_model(X_valid_tensor)
        validation_probs = torch.sigmoid(validation_logits)
        validation_preds = (validation_probs >= 0.5).int().cpu().numpy()

    validation_accuracy = accuracy_score(
        y_valid,
        validation_preds.ravel(),
    )

    history["loss"].append(average_loss)
    history["validation_accuracy"].append(validation_accuracy)

    # Early stopping based on validation accuracy
    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in nn_model.state_dict().items()
        }
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {average_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

# Restore best validation checkpoint
if best_state is not None:
    nn_model.load_state_dict(best_state)

# Final neural network predictions
nn_model.eval()

with torch.no_grad():
    final_logits = nn_model(X_valid_tensor)
    final_probs = torch.sigmoid(final_logits)
    nn_predictions = (final_probs >= 0.5).int().cpu().numpy().ravel()

nn_accuracy = accuracy_score(y_valid, nn_predictions)

print("\nPyTorch Neural Network Accuracy:", round(nn_accuracy, 4))
print("\nPyTorch Classification Report:")
print(classification_report(y_valid, nn_predictions))

print("PyTorch Confusion Matrix:")
print(confusion_matrix(y_valid, nn_predictions))

# -------------------- 9. Plot Training History --------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["loss"])
axes[0].set_title("PyTorch Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

axes[1].plot(history["validation_accuracy"])
axes[1].set_title("PyTorch Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")

plt.tight_layout()
plt.show()

# -------------------- 10. Compare Models --------------------

comparison = pd.DataFrame({
    "Model": ["Random Forest", "PyTorch Neural Network"],
    "Validation Accuracy": [rf_accuracy, nn_accuracy],
})

print("\nModel Comparison:")
print(comparison.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.bar(
    comparison["Model"],
    comparison["Validation Accuracy"],
)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Titanic Model Comparison")
plt.tight_layout()
plt.show()

best_model_name = comparison.loc[
    comparison["Validation Accuracy"].idxmax(),
    "Model",
]

print("\nBest model on this validation split:", best_model_name)
print("\nProject completed successfully!")