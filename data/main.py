
# ==============================================================
# TITANIC MACHINE LEARNING LAB
# Pandas | NumPy | Matplotlib | Scikit-learn | PyTorch
# ==============================================================

import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


# ==============================================================
# 1. CONFIGURATION
# ==============================================================

SEED = 42
EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 32

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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

PALETTE = [
    COLORS["purple"],
    COLORS["blue"],
    COLORS["cyan"],
    COLORS["pink"],
    COLORS["green"],
    COLORS["orange"],
]

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
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "savefig.facecolor": COLORS["background"],
})

print("=" * 68)
print("TITANIC MACHINE LEARNING LAB")
print("=" * 68)
print("Device:", DEVICE)


# ==============================================================
# 2. PLOTTING UTILITIES
# ==============================================================

def style_axis(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", pad=12)

    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(
        axis="y",
        color=COLORS["grid"],
        alpha=0.55,
        linewidth=0.8,
    )
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COLORS["grid"])


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    print(f"Saved chart: {path}")


def add_bar_labels(ax, bars, percent=False):
    for bar in bars:
        value = bar.get_height()
        label = f"{value:.1%}" if percent else f"{value:.3f}"

        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=COLORS["text"],
            fontsize=9,
            fontweight="bold",
        )


# ==============================================================
# 3. LOAD DATA
# ==============================================================

data_path = Path("data/train.csv")

if not data_path.exists():
    raise FileNotFoundError(
        "File not found: data/train.csv\n"
        "Place train.csv inside the project's data folder."
    )

train = pd.read_csv(data_path)

required_columns = {
    "PassengerId", "Survived", "Pclass", "Name", "Sex",
    "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
}

missing_columns = required_columns - set(train.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {sorted(missing_columns)}")

print("\nDataset shape:", train.shape)
print("\nFirst five rows:")
print(train.head())

print("\nColumn data types:")
print(train.dtypes)

print("\nMissing values:")
print(train.isnull().sum())

print("\nTarget distribution:")
print(train["Survived"].value_counts().sort_index())

print("\nTarget proportions:")
print(train["Survived"].value_counts(normalize=True).sort_index().round(3))


# ==============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ==============================================================

survival_by_class = train.groupby("Pclass")["Survived"].mean()
survival_by_sex = train.groupby("Sex")["Survived"].mean()

print("\nSurvival rate by class:")
print(survival_by_class.round(3))

print("\nSurvival rate by sex:")
print(survival_by_sex.round(3))

class_rates = survival_by_class.to_numpy(dtype=np.float64)

print("\nMean of class survival rates:", round(float(np.mean(class_rates)), 4))
print("Standard deviation:", round(float(np.std(class_rates)), 4))

# Four-panel EDA dashboard
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.subplots_adjust(
    left=0.09, right=0.96, top=0.83, bottom=0.10,
    hspace=0.42, wspace=0.25,
)

fig.suptitle(
    "TITANIC | EXPLORATORY DATA ANALYSIS",
    x=0.09, y=0.96, ha="left", fontsize=20,
)

fig.text(
    0.09, 0.91,
    "Passenger demographics and survival patterns",
    color=COLORS["muted"],
)

# Survival by class
ax = axes[0, 0]
bars = ax.bar(
    survival_by_class.index.astype(str),
    survival_by_class.values,
    color=[COLORS["purple"], COLORS["blue"], COLORS["cyan"]],
    width=0.58,
)

style_axis(ax, "Survival Rate by Class", "Passenger class", "Survival rate")
ax.set_ylim(0, 1.12)
add_bar_labels(ax, bars, percent=True)

# Survival by sex
ax = axes[0, 1]
sex_rates = survival_by_sex.sort_values(ascending=False)

bars = ax.bar(
    sex_rates.index,
    sex_rates.values,
    color=[
        COLORS["pink"] if sex == "female" else COLORS["blue"]
        for sex in sex_rates.index
    ],
    width=0.58,
)

style_axis(ax, "Survival Rate by Sex", "Sex", "Survival rate")
ax.set_ylim(0, 1.12)
add_bar_labels(ax, bars, percent=True)

# Age distribution
ax = axes[1, 0]
ages = train["Age"].dropna()

ax.hist(
    ages,
    bins=25,
    color=COLORS["cyan"],
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

style_axis(ax, "Passenger Age Distribution", "Age", "Number of passengers")
ax.legend(frameon=False, labelcolor=COLORS["text"])

# Survival by age group
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
    color=PALETTE[:len(age_survival)],
    width=0.65,
)

style_axis(ax, "Survival Rate by Age Group", "Age group", "Survival rate")
ax.set_ylim(0, 1.12)
ax.tick_params(axis="x", rotation=12)
add_bar_labels(ax, bars, percent=True)

save_figure(fig, "01_eda_dashboard.png")
plt.show()


# ==============================================================
# 5. FEATURE ENGINEERING
# ==============================================================

def create_features(data):
    data = data.copy()

    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    data["Title"] = data["Name"].str.extract(
        r",\s*([^.]*)\.",
        expand=False,
    )

    data["Title"] = data["Title"].fillna("Unknown").str.strip()

    common_titles = ["Mr", "Miss", "Mrs", "Master"]

    data.loc[
        ~data["Title"].isin(common_titles),
        "Title",
    ] = "Other"

    data["Deck"] = (
        data["Cabin"].astype("string").str[0].fillna("Unknown")
    )

    return data


train_features = create_features(train)

FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "Embarked", "FamilySize", "IsAlone", "Title", "Deck",
]

NUMERIC_FEATURES = [
    "Pclass", "Age", "SibSp", "Parch", "Fare",
    "FamilySize", "IsAlone",
]

CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title", "Deck"]

X = train_features[FEATURES]
y = train_features["Survived"].astype("int64")

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y,
)

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_valid))


# ==============================================================
# 6. PREPROCESSING FOR CLASSICAL MODELS
# ==============================================================

preprocessor = ColumnTransformer([
    (
        "numeric",
        SimpleImputer(strategy="median"),
        NUMERIC_FEATURES,
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]),
        CATEGORICAL_FEATURES,
    ),
])

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_valid_processed = preprocessor.transform(X_valid)

if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()

if hasattr(X_valid_processed, "toarray"):
    X_valid_processed = X_valid_processed.toarray()

X_train_processed = np.asarray(X_train_processed, dtype=np.float64)
X_valid_processed = np.asarray(X_valid_processed, dtype=np.float64)

# Scale features for distance-based models and neural networks
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_processed)
X_valid_scaled = scaler.transform(X_valid_processed)


# ==============================================================
# 7. TRAIN AND COMPARE CLASSICAL MODELS
# ==============================================================

models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        "scaled",
    ),
    "KNN": (
        KNeighborsClassifier(n_neighbors=7),
        "scaled",
    ),
    "SVM (RBF)": (
        SVC(C=1.0, kernel="rbf", probability=True, random_state=SEED),
        "scaled",
    ),
    "Decision Tree": (
        DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=5,
            random_state=SEED,
        ),
        "raw",
    ),
    "Random Forest": (
        RandomForestClassifier(
            n_estimators=300,
            random_state=SEED,
            class_weight="balanced",
        ),
        "raw",
    ),
    "Extra Trees": (
        ExtraTreesClassifier(
            n_estimators=300,
            random_state=SEED,
            class_weight="balanced",
        ),
        "raw",
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=SEED),
        "raw",
    ),
    "Hist Gradient Boosting": (
        HistGradientBoostingClassifier(
            max_iter=150,
            random_state=SEED,
        ),
        "raw",
    ),
    "AdaBoost": (
        AdaBoostClassifier(n_estimators=100, random_state=SEED),
        "raw",
    ),
    "Gaussian Naive Bayes": (
        GaussianNB(),
        "raw",
    ),
    "MLP (Scikit-learn)": (
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=700,
            early_stopping=True,
            random_state=SEED,
        ),
        "scaled",
    ),
}

results = []
trained_models = {}

print("\n" + "=" * 60)
print("TRAINING CLASSICAL MODELS")
print("=" * 60)

for name, (estimator, representation) in models.items():
    print(f"\nTraining {name}...")

    if representation == "scaled":
        Xtr, Xva = X_train_scaled, X_valid_scaled
    else:
        Xtr, Xva = X_train_processed, X_valid_processed

    model = clone(estimator)
    model.fit(Xtr, y_train)

    predictions = model.predict(Xva)

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_valid, predictions),
        "Precision": precision_score(
            y_valid, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_valid, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_valid, predictions, zero_division=0
        ),
    }

    results.append(row)

    trained_models[name] = {
        "model": model,
        "representation": representation,
        "predictions": predictions,
    }

    print(
        f"Accuracy: {row['Accuracy']:.4f} | "
        f"F1: {row['F1 Score']:.4f}"
    )

results_df = pd.DataFrame(results)

print("\nCLASSICAL MODEL RESULTS")
print(
    results_df.sort_values("F1 Score", ascending=False)
    .round(4)
    .to_string(index=False)
)

results_df.to_csv(
    OUTPUT_DIR / "classical_model_results.csv",
    index=False,
)

# Keep these variables available for final comparison
rf_predictions = trained_models["Random Forest"]["predictions"]
rf_accuracy = accuracy_score(y_valid, rf_predictions)

print("\nRandom Forest classification report:")
print(classification_report(y_valid, rf_predictions, zero_division=0))
print("Random Forest confusion matrix:")
print(confusion_matrix(y_valid, rf_predictions))


# ==============================================================
# 8. PREPARE DATA FOR PYTORCH
# ==============================================================

torch_preprocessor = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]),
        NUMERIC_FEATURES,
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]),
        CATEGORICAL_FEATURES,
    ),
])

X_train_torch = torch_preprocessor.fit_transform(X_train)
X_valid_torch = torch_preprocessor.transform(X_valid)

if hasattr(X_train_torch, "toarray"):
    X_train_torch = X_train_torch.toarray()

if hasattr(X_valid_torch, "toarray"):
    X_valid_torch = X_valid_torch.toarray()

X_train_torch = np.asarray(X_train_torch, dtype=np.float32)
X_valid_torch = np.asarray(X_valid_torch, dtype=np.float32)

y_train_torch = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)

X_train_tensor = torch.from_numpy(X_train_torch)
y_train_tensor = torch.from_numpy(y_train_torch)
X_valid_tensor = torch.from_numpy(X_valid_torch).to(DEVICE)


# ==============================================================
# 9. PYTORCH NEURAL NETWORK
# ==============================================================

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
    batch_size=BATCH_SIZE,
    shuffle=True,
)

history = {
    "loss": [],
    "validation_accuracy": [],
}

best_accuracy = -1.0
best_state = None
epochs_without_improvement = 0

print("\n" + "=" * 60)
print("TRAINING PYTORCH NEURAL NETWORK")
print("=" * 60)

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

        validation_predictions = (
            (validation_probs >= 0.5)
            .int()
            .cpu()
            .numpy()
            .ravel()
        )

    validation_accuracy = accuracy_score(
        y_valid, validation_predictions
    )

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

nn_result = {
    "Model": "PyTorch Neural Network",
    "Accuracy": nn_accuracy,
    "Precision": precision_score(
        y_valid, nn_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_valid, nn_predictions, zero_division=0
    ),
    "F1 Score": f1_score(
        y_valid, nn_predictions, zero_division=0
    ),
}

print("\nPyTorch accuracy:", f"{nn_accuracy:.4f}")
print("\nPyTorch classification report:")
print(classification_report(y_valid, nn_predictions, zero_division=0))
print("PyTorch confusion matrix:")
print(confusion_matrix(y_valid, nn_predictions))


# ==============================================================
# 10. PYTORCH TRAINING DASHBOARD
# ==============================================================

epochs = np.arange(1, len(history["loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.subplots_adjust(
    left=0.08, right=0.96, top=0.80, bottom=0.15, wspace=0.28
)

fig.suptitle(
    "PYTORCH | TRAINING PERFORMANCE",
    x=0.08, y=0.95, ha="left", fontsize=19,
)

fig.text(
    0.08, 0.88,
    "Optimization and validation monitoring",
    color=COLORS["muted"],
)

ax = axes[0]
ax.plot(
    epochs, history["loss"],
    color=COLORS["purple"], linewidth=2.8,
)
ax.fill_between(
    epochs, history["loss"],
    color=COLORS["purple"], alpha=0.13,
)
style_axis(ax, "Training Loss", "Epoch", "Binary cross-entropy loss")

ax = axes[1]
ax.plot(
    epochs, history["validation_accuracy"],
    color=COLORS["green"], linewidth=2.8,
)
ax.fill_between(
    epochs, history["validation_accuracy"],
    color=COLORS["green"], alpha=0.13,
)
ax.axhline(
    best_accuracy,
    color=COLORS["orange"],
    linestyle="--",
    linewidth=1.5,
    label=f"Best: {best_accuracy:.1%}",
)
style_axis(ax, "Validation Accuracy", "Epoch", "Accuracy")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, labelcolor=COLORS["text"])

save_figure(fig, "02_pytorch_training.png")
plt.show()


# ==============================================================
# 11. FINAL MODEL LEADERBOARD
# ==============================================================

all_results = pd.concat(
    [
        results_df,
        pd.DataFrame([nn_result]),
    ],
    ignore_index=True,
)

all_results = all_results.sort_values(
    "F1 Score", ascending=True
).reset_index(drop=True)

all_results.to_csv(
    OUTPUT_DIR / "all_model_results.csv",
    index=False,
)

print("\n" + "=" * 75)
print("FINAL MODEL LEADERBOARD")
print("=" * 75)

print(
    all_results.sort_values("F1 Score", ascending=False)
    .round(4)
    .to_string(index=False)
)

# Chart A: F1 leaderboard
fig, ax = plt.subplots(figsize=(12, 8))
fig.subplots_adjust(left=0.30, right=0.94, top=0.88, bottom=0.12)

bar_colors = []

for name in all_results["Model"]:
    if name == "PyTorch Neural Network":
        bar_colors.append(COLORS["purple"])
    elif name == "Random Forest":
        bar_colors.append(COLORS["cyan"])
    else:
        bar_colors.append(COLORS["blue"])

bars = ax.barh(
    all_results["Model"],
    all_results["F1 Score"],
    color=bar_colors,
    height=0.68,
)

style_axis(ax, "MODEL LEADERBOARD", "Model", "F1 score for survivors")
ax.set_xlim(0, 1.0)
ax.grid(axis="x", color=COLORS["grid"], alpha=0.55)
ax.grid(axis="y", visible=False)

for bar, score in zip(bars, all_results["F1 Score"]):
    ax.text(
        min(score + 0.015, 0.94),
        bar.get_y() + bar.get_height() / 2,
        f"{score:.3f}",
        va="center",
        color=COLORS["text"],
        fontweight="bold",
    )

save_figure(fig, "03_model_f1_leaderboard.png")
plt.show()

# Chart B: Compare metrics for every model
metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score"]

fig, ax = plt.subplots(figsize=(14, 8))
fig.subplots_adjust(left=0.30, right=0.96, top=0.88, bottom=0.12)

y_positions = np.arange(len(all_results))
bar_height = 0.19

metric_colors = [
    COLORS["blue"],
    COLORS["pink"],
    COLORS["orange"],
    COLORS["green"],
]

for index, (metric, color) in enumerate(
    zip(metric_columns, metric_colors)
):
    ax.barh(
        y_positions + (index - 1.5) * bar_height,
        all_results[metric],
        height=bar_height,
        color=color,
        label=metric,
    )

ax.set_yticks(y_positions)
ax.set_yticklabels(all_results["Model"])
ax.set_xlim(0, 1.0)
ax.set_xlabel("Score")
ax.set_title("ALL MODELS | METRIC COMPARISON", loc="left", pad=14)
ax.grid(axis="x", color=COLORS["grid"], alpha=0.55)
ax.set_axisbelow(True)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(COLORS["grid"])

ax.legend(
    loc="lower right",
    frameon=False,
    labelcolor=COLORS["text"],
    ncol=2,
)

save_figure(fig, "04_all_metrics.png")
plt.show()


# ==============================================================
# 12. CONFUSION MATRIX FOR THE TOP MODEL
# ==============================================================

best_name = all_results.loc[
    all_results["F1 Score"].idxmax(), "Model"
]

if best_name == "PyTorch Neural Network":
    best_predictions = nn_predictions
else:
    best_predictions = trained_models[best_name]["predictions"]

cm = confusion_matrix(y_valid, best_predictions, labels=[0, 1])

fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(
    cm,
    cmap="magma",
    interpolation="nearest",
)

ax.set_title(f"CONFUSION MATRIX | {best_name}", loc="left", pad=16)
ax.set_xlabel("Predicted label")
ax.set_ylabel("Actual label")
ax.set_xticks([0, 1], labels=["Did not survive", "Survived"])
ax.set_yticks([0, 1], labels=["Did not survive", "Survived"])

threshold = cm.max() / 2 if cm.size else 0

for row in range(cm.shape[0]):
    for col in range(cm.shape[1]):
        ax.text(
            col,
            row,
            str(cm[row, col]),
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="white" if cm[row, col] > threshold else "black",
        )

fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

save_figure(fig, "05_best_model_confusion_matrix.png")
plt.show()


# ==============================================================
# 13. SUMMARY
# ==============================================================

best_row = all_results.loc[all_results["F1 Score"].idxmax()]

print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)
print("Number of classical models:", len(models))
print("Number of models compared:", len(all_results))
print("Top model by survivor F1:", best_row["Model"])
print(f"Accuracy:  {best_row['Accuracy']:.4f}")
print(f"Precision: {best_row['Precision']:.4f}")
print(f"Recall:    {best_row['Recall']:.4f}")
print(f"F1 score:  {best_row['F1 Score']:.4f}")

print("\nOutputs saved in:", OUTPUT_DIR.resolve())
print("Project completed successfully!")