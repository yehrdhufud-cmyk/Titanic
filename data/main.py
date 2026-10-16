
# ============================================================
# TITANIC SURVIVAL PREDICTION
# Pandas | NumPy | Matplotlib | Scikit-learn | PyTorch
# ============================================================

import random
from pathlib import Path

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
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLORS = {
    "background": "#0B1120",
    "panel": "#151F32",
    "text": "#F1F5F9",
    "muted": "#94A3B8",
    "purple": "#A78BFA",
    "blue": "#60A5FA",
    "cyan": "#22D3EE",
    "pink": "#F472B6",
    "green": "#34D399",
    "orange": "#FBBF24",
    "grid": "#334155",
}

plt.rcParams.update({
    "figure.facecolor": COLORS["background"],
    "axes.facecolor": COLORS["panel"],
    "axes.edgecolor": COLORS["grid"],
    "axes.labelcolor": COLORS["muted"],
    "axes.titlecolor": COLORS["text"],
    "xtick.color": COLORS["muted"],
    "ytick.color": COLORS["muted"],
    "text.color": COLORS["text"],
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "savefig.facecolor": COLORS["background"],
})

print("=" * 60)
print("TITANIC SURVIVAL PREDICTION")
print("=" * 60)
print("Device:", DEVICE)


# -------------------- 2. Plot helpers --------------------

def style_axis(ax, title, ylabel=None, xlabel=None):
    ax.set_title(title, loc="left", pad=14)

    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)

    ax.grid(
        axis="y",
        color=COLORS["grid"],
        alpha=0.55,
        linewidth=0.8,
    )
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])


def add_bar_labels(ax, bars, percent=False):
    for bar in bars:
        value = bar.get_height()
        label = f"{value:.1%}" if percent else f"{value:.2f}"

        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=COLORS["text"],
            fontsize=9,
            fontweight="bold",
        )


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    print(f"Chart saved: {path}")


# -------------------- 3. Load data --------------------

data_path = Path("data/train.csv")

if not data_path.exists():
    raise FileNotFoundError(
        "data/train.csv not found. Put train.csv inside the data folder."
    )

train = pd.read_csv(data_path)

required_columns = {
    "PassengerId", "Survived", "Pclass", "Name", "Sex",
    "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
}

missing_columns = required_columns - set(train.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns in train.csv: {sorted(missing_columns)}"
    )

print("\nDataset shape:", train.shape)
print("\nFirst five rows:")
print(train.head())

print("\nMissing values:")
print(train.isnull().sum())

print("\nSurvival counts:")
print(train["Survived"].value_counts().sort_index())


# -------------------- 4. Exploratory data analysis --------------------

survival_by_class = train.groupby("Pclass")["Survived"].mean()
survival_by_sex = train.groupby("Sex")["Survived"].mean()

print("\nSurvival rate by class:")
print(survival_by_class.round(3))

print("\nSurvival rate by sex:")
print(survival_by_sex.round(3))

# NumPy numerical summary
class_rates = survival_by_class.to_numpy(dtype=np.float64)

print("\nMean class survival rate:", round(float(np.mean(class_rates)), 4))
print("Standard deviation:", round(float(np.std(class_rates)), 4))

# Four-panel EDA dashboard
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.subplots_adjust(
    left=0.09,
    right=0.96,
    top=0.83,
    bottom=0.10,
    hspace=0.42,
    wspace=0.25,
)

fig.suptitle(
    "TITANIC  |  EXPLORATORY DATA ANALYSIS",
    x=0.09,
    y=0.96,
    ha="left",
    fontsize=20,
    fontweight="bold",
)

fig.text(
    0.09,
    0.91,
    "Understanding passenger survival patterns",
    color=COLORS["muted"],
    fontsize=11,
)

# Chart 1: Survival by class
ax = axes[0, 0]

bars = ax.bar(
    survival_by_class.index.astype(str),
    survival_by_class.values,
    color=[COLORS["purple"], COLORS["blue"], COLORS["cyan"]],
    width=0.58,
    edgecolor="none",
)

style_axis(ax, "Survival Rate by Class", "Survival rate", "Passenger class")
ax.set_ylim(0, 1.12)
add_bar_labels(ax, bars, percent=True)

# Chart 2: Survival by sex
ax = axes[0, 1]
sex_rates = survival_by_sex.sort_values(ascending=False)

bars = ax.bar(
    sex_rates.index,
    sex_rates.values,
    color=[
        COLORS["pink"] if sex == "female" else COLORS["blue"]
        for sex in sex_rates.index
    ],
    width=0.55,
    edgecolor="none",
)

style_axis(ax, "Survival Rate by Sex", "Survival rate", "Sex")
ax.set_ylim(0, 1.12)
add_bar_labels(ax, bars, percent=True)

# Chart 3: Age distribution
ax = axes[1, 0]
ages = train["Age"].dropna()

ax.hist(
    ages,
    bins=25,
    color=COLORS["cyan"],
    alpha=0.88,
    edgecolor=COLORS["panel"],
    linewidth=1,
)

ax.axvline(
    ages.median(),
    color=COLORS["orange"],
    linestyle="--",
    linewidth=2,
    label=f"Median: {ages.median():.1f}",
)

style_axis(ax, "Passenger Age Distribution", "Passengers", "Age")
ax.legend(frameon=False, labelcolor=COLORS["text"])

# Chart 4: Survival by age group
ax = axes[1, 1]
age_data = train[["Age", "Survived"]].dropna().copy()

age_data["Age Group"] = pd.cut(
    age_data["Age"],
    bins=[0, 12, 18, 35, 60, np.inf],
    labels=["Child", "Teen", "Young adult", "Adult", "Senior"],
    include_lowest=True,
)

age_survival = age_data.groupby(
    "Age Group", observed=False
)["Survived"].mean()

bars = ax.bar(
    age_survival.index.astype(str),
    age_survival.values,
    color=[
        COLORS["green"],
        COLORS["cyan"],
        COLORS["blue"],
        COLORS["purple"],
        COLORS["pink"],
    ],
    width=0.65,
)

style_axis(ax, "Survival Rate by Age Group", "Survival rate", "Age group")
ax.set_ylim(0, 1.12)
ax.tick_params(axis="x", rotation=12)
add_bar_labels(ax, bars, percent=True)

save_figure(fig, "titanic_eda_dashboard.png")
plt.show()


# -------------------- 5. Feature engineering --------------------

def create_features(data):
    data = data.copy()

    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    # Extract title from passenger name
    data["Title"] = data["Name"].str.extract(
        r",\s*([^.]*)\.",
        expand=False,
    )

    data["Title"] = data["Title"].fillna("Unknown").str.strip()

    common_titles = ["Mr", "Miss", "Mrs", "Master"]
    data.loc[
        ~data["Title"].isin(common_titles),
        "Title"
    ] = "Other"

    # Cabin deck, including missing values
    data["Deck"] = (
        data["Cabin"].astype("string").str[0].fillna("Unknown")
    )

    return data


train_features = create_features(train)

features = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "Embarked", "FamilySize", "IsAlone", "Title", "Deck",
]

numeric_features = [
    "Pclass", "Age", "SibSp", "Parch", "Fare",
    "FamilySize", "IsAlone",
]

categorical_features = ["Sex", "Embarked", "Title", "Deck"]

X = train_features[features]
y = train_features["Survived"].astype("int64")

# Split before fitting preprocessing to prevent leakage
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y,
)


# -------------------- 6. Scikit-learn Random Forest --------------------

sklearn_preprocessor = ColumnTransformer([
    (
        "numeric",
        SimpleImputer(strategy="median"),
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

rf_model = Pipeline([
    ("preprocessor", sklearn_preprocessor),
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

print("\nRandom Forest accuracy:", f"{rf_accuracy:.4f}")
print("\nRandom Forest classification report:")
print(classification_report(y_valid, rf_predictions, zero_division=0))
print("Random Forest confusion matrix:")
print(confusion_matrix(y_valid, rf_predictions))


# -------------------- 7. PyTorch preprocessing --------------------

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

# Convert sparse matrices for PyTorch
if hasattr(X_train_np, "toarray"):
    X_train_np = X_train_np.toarray()

if hasattr(X_valid_np, "toarray"):
    X_valid_np = X_valid_np.toarray()

X_train_np = np.asarray(X_train_np, dtype=np.float32)
X_valid_np = np.asarray(X_valid_np, dtype=np.float32)

y_train_np = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)

X_train_tensor = torch.from_numpy(X_train_np)
y_train_tensor = torch.from_numpy(y_train_np)
X_valid_tensor = torch.from_numpy(X_valid_np).to(DEVICE)


# -------------------- 8. Build neural network --------------------

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


nn_model = TitanicNet(X_train_tensor.shape[1]).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    nn_model.parameters(),
    lr=0.001,
    weight_decay=1e-4,
)

train_dataset = torch.utils.data.TensorDataset(
    X_train_tensor,
    y_train_tensor,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)


# -------------------- 9. Train PyTorch model --------------------

EPOCHS = 100
PATIENCE = 15

best_accuracy = -1.0
best_state = None
epochs_without_improvement = 0

history = {
    "loss": [],
    "validation_accuracy": [],
}

print("\nTraining PyTorch neural network...")

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

        validation_preds = (
            (validation_probs >= 0.5)
            .int()
            .cpu()
            .numpy()
            .ravel()
        )

    validation_accuracy = accuracy_score(y_valid, validation_preds)

    history["loss"].append(average_loss)
    history["validation_accuracy"].append(validation_accuracy)

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
            f"Validation accuracy: {validation_accuracy:.4f}"
        )

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch + 1}")
        break

if best_state is not None:
    nn_model.load_state_dict(best_state)

nn_model.eval()

with torch.no_grad():
    final_logits = nn_model(X_valid_tensor)
    final_probs = torch.sigmoid(final_logits)

    nn_predictions = (
        (final_probs >= 0.5)
        .int()
        .cpu()
        .numpy()
        .ravel()
    )

nn_accuracy = accuracy_score(y_valid, nn_predictions)

print("\nPyTorch neural network accuracy:", f"{nn_accuracy:.4f}")
print("\nPyTorch classification report:")
print(classification_report(y_valid, nn_predictions, zero_division=0))
print("PyTorch confusion matrix:")
print(confusion_matrix(y_valid, nn_predictions))


# -------------------- 10. Neural network training dashboard --------------------

epochs = np.arange(1, len(history["loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.subplots_adjust(
    left=0.08,
    right=0.96,
    top=0.80,
    bottom=0.15,
    wspace=0.28,
)

fig.suptitle(
    "PYTORCH  |  TRAINING PERFORMANCE",
    x=0.08,
    y=0.95,
    ha="left",
    fontsize=19,
)

fig.text(
    0.08,
    0.88,
    "Monitoring optimization and validation results",
    color=COLORS["muted"],
    fontsize=10,
)

# Loss
ax = axes[0]

ax.plot(
    epochs,
    history["loss"],
    color=COLORS["purple"],
    linewidth=2.8,
)

ax.fill_between(
    epochs,
    history["loss"],
    color=COLORS["purple"],
    alpha=0.13,
)

style_axis(ax, "Training Loss", "Binary cross-entropy loss", "Epoch")

# Validation accuracy
ax = axes[1]

ax.plot(
    epochs,
    history["validation_accuracy"],
    color=COLORS["green"],
    linewidth=2.8,
)

ax.fill_between(
    epochs,
    history["validation_accuracy"],
    color=COLORS["green"],
    alpha=0.13,
)

ax.axhline(
    best_accuracy,
    color=COLORS["orange"],
    linestyle="--",
    linewidth=1.5,
    label=f"Best: {best_accuracy:.1%}",
)

style_axis(ax, "Validation Accuracy", "Accuracy", "Epoch")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, labelcolor=COLORS["text"])

save_figure(fig, "titanic_training_dashboard.png")
plt.show()


# -------------------- 11. Model comparison dashboard --------------------

comparison = pd.DataFrame({
    "Model": ["Random Forest", "PyTorch Neural Network"],
    "Validation Accuracy": [rf_accuracy, nn_accuracy],
})

print("\nModel comparison:")
print(
    comparison.to_string(
        index=False,
        formatters={
            "Validation Accuracy": lambda value: f"{value:.2%}"
        },
    )
)

fig, ax = plt.subplots(figsize=(10, 5.5))

bars = ax.barh(
    comparison["Model"],
    comparison["Validation Accuracy"],
    color=[COLORS["blue"], COLORS["purple"]],
    height=0.48,
    edgecolor="none",
)

style_axis(
    ax,
    "Which Model Performs Better?",
    "Model",
    "Validation accuracy",
)

ax.set_xlim(0, 1.0)
ax.set_xticks(np.arange(0, 1.01, 0.1))
ax.grid(axis="x", color=COLORS["grid"], alpha=0.55)
ax.grid(axis="y", visible=False)

for bar, score in zip(bars, comparison["Validation Accuracy"]):
    ax.text(
        min(score + 0.025, 0.94),
        bar.get_y() + bar.get_height() / 2,
        f"{score:.1%}",
        va="center",
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=COLORS["text"],
    )

fig.text(
    0.12,
    0.04,
    "Same validation split for both models; higher accuracy is better.",
    color=COLORS["muted"],
    fontsize=9,
)

save_figure(fig, "titanic_model_comparison.png")
plt.show()

best_model_name = comparison.loc[
    comparison["Validation Accuracy"].idxmax(),
    "Model",
]

print("\nBest model on this validation split:", best_model_name)
print("Charts saved to:", OUTPUT_DIR.resolve())
print("\nProject completed successfully!")