
from pathlib import Path
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

warnings.filterwarnings("ignore", category=FutureWarning)

# PyTorch is optional
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
VALID_SIZE = 0.20
EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 32

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

if TORCH_AVAILABLE:
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
else:
    DEVICE = None


# Black background + bright chart colors
COLORS = {
    "background": "#05070D",
    "panel": "#101521",
    "text": "#F3F4F6",
    "muted": "#A8B0C0",
    "grid": "#293244",
    "blue": "#60A5FA",
    "cyan": "#22D3EE",
    "purple": "#C4B5FD",
    "pink": "#FB7185",
    "green": "#34D399",
    "gold": "#FBBF24",
    "orange": "#FB923C",
}

PALETTE = [
    COLORS["blue"],
    COLORS["cyan"],
    COLORS["purple"],
    COLORS["pink"],
    COLORS["green"],
    COLORS["gold"],
]

plt.rcParams.update({
    "figure.facecolor": COLORS["background"],
    "axes.facecolor": COLORS["panel"],
    "axes.edgecolor": COLORS["grid"],
    "axes.labelcolor": COLORS["text"],
    "axes.titlecolor": COLORS["text"],
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "xtick.color": COLORS["muted"],
    "ytick.color": COLORS["muted"],
    "text.color": COLORS["text"],
    "grid.color": COLORS["grid"],
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "savefig.facecolor": COLORS["background"],
    "savefig.edgecolor": COLORS["background"],
    "savefig.dpi": 220,
})


# ============================================================
# 2. PLOTTING FUNCTIONS
# ============================================================

def style_axis(
    ax,
    title=None,
    xlabel=None,
    ylabel=None,
    grid_axis="y"
):
    ax.set_facecolor(COLORS["panel"])

    if title:
        ax.set_title(title, loc="left", pad=12)

    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(
        True,
        axis=grid_axis,
        color=COLORS["grid"],
        alpha=0.8,
        linewidth=0.8,
    )

    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["grid"])

    ax.tick_params(
        colors=COLORS["muted"],
        length=0,
        pad=6,
    )


def save_figure(fig, filename):
    fig.patch.set_facecolor(COLORS["background"])

    for ax in fig.axes:
        ax.set_facecolor(COLORS["panel"])
        ax.tick_params(colors=COLORS["muted"])
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])
        ax.title.set_color(COLORS["text"])

        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])

        legend = ax.get_legend()

        if legend:
            legend.get_frame().set_facecolor(COLORS["panel"])
            legend.get_frame().set_edgecolor(COLORS["grid"])

            for item in legend.get_texts():
                item.set_color(COLORS["text"])

    fig.tight_layout(pad=1.8)

    path = OUTPUT_DIR / filename

    fig.savefig(
        path,
        bbox_inches="tight",
        facecolor=COLORS["background"],
    )

    print(f"Saved chart: {path}")

    plt.show()
    plt.close(fig)


def add_bar_labels(ax, bars, percent=False):
    for bar in bars:
        value = bar.get_height()

        label = (
            f"{value:.1%}"
            if percent
            else f"{value:.3f}"
        )

        ax.annotate(
            label,
            (
                bar.get_x() + bar.get_width() / 2,
                value,
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=COLORS["text"],
            fontsize=9,
            fontweight="bold",
        )


def make_one_hot_encoder():
    # Support old and new scikit-learn versions
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


print("=" * 68)
print("TITANIC MACHINE LEARNING LAB")
print("=" * 68)
print("Project folder:", BASE_DIR)

if TORCH_AVAILABLE:
    print("PyTorch device:", DEVICE)
else:
    print("PyTorch is not installed; its model will be skipped.")


# ============================================================
# 3. LOAD DATA
# ============================================================

train_path = BASE_DIR / "train.csv"
test_path = BASE_DIR / "test.csv"

if not train_path.exists():
    raise FileNotFoundError(
        f"Cannot find {train_path}\n"
        "Place train.csv beside main.py."
    )

if not test_path.exists():
    raise FileNotFoundError(
        f"Cannot find {test_path}\n"
        "Place test.csv beside main.py."
    )

train_raw = pd.read_csv(train_path)
test_raw = pd.read_csv(test_path)

required_columns = {
    "PassengerId", "Survived", "Pclass", "Name",
    "Sex", "Age", "SibSp", "Parch", "Ticket",
    "Fare", "Cabin", "Embarked",
}

missing_columns = required_columns - set(train_raw.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns in train.csv: {sorted(missing_columns)}"
    )

if "PassengerId" not in test_raw.columns:
    raise ValueError("test.csv must contain PassengerId.")

print("\nTraining shape:", train_raw.shape)
print("Test shape:", test_raw.shape)

print("\nFirst five rows:")
print(train_raw.head().to_string(index=False))

print("\nMissing values:")
print(train_raw.isnull().sum().to_string())

print("\nTarget distribution:")
print(train_raw["Survived"].value_counts().sort_index())


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

survival_by_class = (
    train_raw.groupby("Pclass")["Survived"].mean()
)

survival_by_sex = (
    train_raw.groupby("Sex")["Survived"]
    .mean()
    .sort_values(ascending=False)
)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

fig.suptitle(
    "TITANIC | EXPLORATORY DATA ANALYSIS",
    x=0.06,
    y=0.98,
    ha="left",
    fontsize=19,
    fontweight="bold",
    color=COLORS["text"],
)

# Survival by passenger class
ax = axes[0, 0]

bars = ax.bar(
    survival_by_class.index.astype(str),
    survival_by_class.values,
    color=[
        COLORS["purple"],
        COLORS["blue"],
        COLORS["cyan"],
    ],
    width=0.62,
)

style_axis(
    ax,
    "Survival rate by passenger class",
    "Passenger class",
    "Survival rate",
)

ax.set_ylim(0, 1.13)
add_bar_labels(ax, bars, percent=True)

# Survival by sex
ax = axes[0, 1]

bars = ax.bar(
    survival_by_sex.index,
    survival_by_sex.values,
    color=[
        COLORS["pink"] if sex == "female"
        else COLORS["blue"]
        for sex in survival_by_sex.index
    ],
    width=0.62,
)

style_axis(
    ax,
    "Survival rate by sex",
    "Sex",
    "Survival rate",
)

ax.set_ylim(0, 1.13)
add_bar_labels(ax, bars, percent=True)

# Age distribution
ax = axes[1, 0]
ages = train_raw["Age"].dropna()

ax.hist(
    ages,
    bins=25,
    color=COLORS["cyan"],
    edgecolor=COLORS["panel"],
    linewidth=1.1,
)

ax.axvline(
    ages.median(),
    color=COLORS["gold"],
    linestyle="--",
    linewidth=2,
    label=f"Median: {ages.median():.1f}",
)

style_axis(
    ax,
    "Passenger age distribution",
    "Age",
    "Passenger count",
)

ax.legend(frameon=True)

# Survival by age group
ax = axes[1, 1]

age_data = train_raw[["Age", "Survived"]].dropna().copy()

age_data["Age group"] = pd.cut(
    age_data["Age"],
    bins=[0, 12, 18, 35, 60, np.inf],
    labels=[
        "Child",
        "Teen",
        "Young adult",
        "Adult",
        "Senior",
    ],
    include_lowest=True,
)

age_rates = age_data.groupby(
    "Age group",
    observed=False,
)["Survived"].mean()

bars = ax.bar(
    age_rates.index.astype(str),
    age_rates.values,
    color=PALETTE[:len(age_rates)],
    width=0.68,
)

style_axis(
    ax,
    "Survival rate by age group",
    "Age group",
    "Survival rate",
)

ax.set_ylim(0, 1.13)
ax.tick_params(axis="x", rotation=12)
add_bar_labels(ax, bars, percent=True)

save_figure(fig, "01_eda_dashboard.png")


# Missing values chart
missing_pct = (
    train_raw.isna().mean() * 100
).sort_values(ascending=False)

missing_pct = missing_pct[missing_pct > 0]

if not missing_pct.empty:
    fig, ax = plt.subplots(figsize=(11, 5))

    bars = ax.bar(
        missing_pct.index,
        missing_pct.values,
        color=COLORS["purple"],
    )

    style_axis(
        ax,
        "Missing values by column",
        "Column",
        "Missing values (%)",
    )

    ax.tick_params(axis="x", rotation=35)
    add_bar_labels(ax, bars)

    save_figure(fig, "02_missing_values.png")


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

def create_features(data):
    data = data.copy()

    # Family-related features
    data["FamilySize"] = (
        data["SibSp"] + data["Parch"] + 1
    )

    data["IsAlone"] = (
        data["FamilySize"] == 1
    ).astype(int)

    # Extract passenger title from their name
    data["Title"] = (
        data["Name"]
        .str.extract(r",\s*([^.]*)\.", expand=False)
        .str.strip()
    )

    data["Title"] = data["Title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    })

    common_titles = {"Mr", "Miss", "Mrs", "Master"}

    data["Title"] = data["Title"].where(
        data["Title"].isin(common_titles),
        "Other",
    )

    # Cabin deck
    data["Deck"] = (
        data["Cabin"]
        .fillna("Unknown")
        .astype(str)
        .str[0]
    )

    # Ticket group size
    data["TicketGroupSize"] = (
        data.groupby("Ticket")["Ticket"]
        .transform("count")
    )

    # Fare per ticket-group member
    data["FarePerPerson"] = (
        data["Fare"]
        / data["TicketGroupSize"].replace(0, 1)
    )

    return data


train = create_features(train_raw)
test = create_features(test_raw)

FEATURES = [
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
    "TicketGroupSize",
    "FarePerPerson",
]

NUMERIC_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "TicketGroupSize",
    "FarePerPerson",
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Title",
    "Deck",
]

X = train[FEATURES]
y = train["Survived"].astype(int)
X_test = test[FEATURES]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=VALID_SIZE,
    random_state=SEED,
    stratify=y,
)

print("\nTraining split:", X_train.shape)
print("Validation split:", X_valid.shape)


# ============================================================
# 6. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]),
        NUMERIC_FEATURES,
    ),
    (
        "categorical",
        Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            ("encoder", make_one_hot_encoder()),
        ]),
        CATEGORICAL_FEATURES,
    ),
])

# Fit preprocessing on training split only
X_train_processed = preprocessor.fit_transform(X_train)
X_valid_processed = preprocessor.transform(X_valid)

X_train_processed = np.asarray(
    X_train_processed,
    dtype=np.float32,
)

X_valid_processed = np.asarray(
    X_valid_processed,
    dtype=np.float32,
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_processed
).astype(np.float32)

X_valid_scaled = scaler.transform(
    X_valid_processed
).astype(np.float32)


# ============================================================
# 7. CLASSICAL MODEL COMPARISON
# ============================================================

models = {
    "Logistic Regression": (
        LogisticRegression(
            max_iter=2000,
            random_state=SEED,
        ),
        "scaled",
    ),

    "KNN": (
        KNeighborsClassifier(n_neighbors=7),
        "scaled",
    ),

    "SVM (RBF)": (
        SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            random_state=SEED,
        ),
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
            min_samples_leaf=2,
        ),
        "raw",
    ),

    "Extra Trees": (
        ExtraTreesClassifier(
            n_estimators=300,
            random_state=SEED,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
        "raw",
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(
            random_state=SEED,
        ),
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
        AdaBoostClassifier(
            n_estimators=100,
            random_state=SEED,
        ),
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

for name, (model, input_type) in models.items():
    print(f"\nTraining {name}...")

    if input_type == "scaled":
        xtr = X_train_scaled
        xva = X_valid_scaled
    else:
        xtr = X_train_processed
        xva = X_valid_processed

    try:
        model.fit(xtr, y_train)
        predictions = model.predict(xva)

        scores = {
            "Model": name,
            "Accuracy": accuracy_score(
                y_valid, predictions
            ),
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

        results.append(scores)
        trained_models[name] = (model, input_type)

        print(
            "F1:", round(scores["F1 Score"], 4),
            "| Accuracy:", round(scores["Accuracy"], 4),
        )

    except Exception as exc:
        print(f"Skipped {name}: {exc}")


# ============================================================
# 8. PYTORCH NEURAL NETWORK
# ============================================================

torch_model = None
torch_scores = None

if TORCH_AVAILABLE:
    print("\nTraining PyTorch Neural Network...")

    class TitanicMLP(nn.Module):
        def __init__(self, input_size):
            super().__init__()

            self.network = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.25),

                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.15),

                nn.Linear(32, 2),
            )

        def forward(self, x):
            return self.network(x)

    Xtr_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
    )

    ytr_tensor = torch.tensor(
        y_train.to_numpy(),
        dtype=torch.long,
    )

    Xva_tensor = torch.tensor(
        X_valid_scaled,
        dtype=torch.float32,
    ).to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(Xtr_tensor, ytr_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    torch_model = TitanicMLP(
        X_train_scaled.shape[1]
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        torch_model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    best_f1 = -1.0
    best_state = None
    patience_left = PATIENCE

    history = {
        "loss": [],
        "val_f1": [],
    }

    for epoch in range(EPOCHS):
        torch_model.train()
        epoch_losses = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            logits = torch_model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        torch_model.eval()

        with torch.no_grad():
            valid_logits = torch_model(Xva_tensor)
            valid_pred = (
                valid_logits.argmax(dim=1)
                .cpu()
                .numpy()
            )

        val_f1 = f1_score(
            y_valid,
            valid_pred,
            zero_division=0,
        )

        history["loss"].append(
            float(np.mean(epoch_losses))
        )

        history["val_f1"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in torch_model.state_dict().items()
            }

            patience_left = PATIENCE
        else:
            patience_left -= 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:03d}/{EPOCHS}"
                f" | Loss: {history['loss'][-1]:.4f}"
                f" | Validation F1: {val_f1:.4f}"
            )

        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    if best_state is not None:
        torch_model.load_state_dict(best_state)

    torch_model.eval()

    with torch.no_grad():
        valid_pred = (
            torch_model(Xva_tensor)
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

    torch_scores = {
        "Model": "PyTorch Neural Network",
        "Accuracy": accuracy_score(
            y_valid, valid_pred
        ),
        "Precision": precision_score(
            y_valid, valid_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_valid, valid_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_valid, valid_pred, zero_division=0
        ),
    }

    results.append(torch_scores)

    print(
        "PyTorch F1:",
        round(torch_scores["F1 Score"], 4),
    )

    # Training history charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(
        history["loss"],
        color=COLORS["cyan"],
        linewidth=2,
    )

    style_axis(
        axes[0],
        "PyTorch training loss",
        "Epoch",
        "Cross-entropy loss",
    )

    axes[1].plot(
        history["val_f1"],
        color=COLORS["purple"],
        linewidth=2,
    )

    style_axis(
        axes[1],
        "Validation F1 by epoch",
        "Epoch",
        "F1 score",
    )

    save_figure(fig, "03_pytorch_training.png")


# ============================================================
# 9. MODEL RESULTS AND COMPARISON CHARTS
# ============================================================

if not results:
    raise RuntimeError(
        "No models trained successfully. Check the errors above."
    )

results_df = (
    pd.DataFrame(results)
    .sort_values("F1 Score", ascending=False)
    .reset_index(drop=True)
)

results_df.to_csv(
    OUTPUT_DIR / "all_model_results.csv",
    index=False,
)

print("\n" + "=" * 68)
print("MODEL LEADERBOARD | VALIDATION F1")
print("=" * 68)

print(results_df.round(4).to_string(index=False))


# F1 leaderboard
fig, ax = plt.subplots(
    figsize=(12, max(6, 0.48 * len(results_df)))
)

plot_df = results_df.sort_values(
    "F1 Score",
    ascending=True,
)

bar_colors = [
    COLORS["purple"]
    if name == "PyTorch Neural Network"
    else COLORS["cyan"]
    if name == "Random Forest"
    else COLORS["blue"]
    for name in plot_df["Model"]
]

bars = ax.barh(
    plot_df["Model"],
    plot_df["F1 Score"],
    color=bar_colors,
    height=0.68,
)

style_axis(
    ax,
    "Model leaderboard | Validation F1",
    "F1 score",
    "Model",
    grid_axis="x",
)

ax.grid(axis="y", visible=False)
ax.set_xlim(0, 1.0)

for bar, score in zip(bars, plot_df["F1 Score"]):
    ax.text(
        min(score + 0.015, 0.95),
        bar.get_y() + bar.get_height() / 2,
        f"{score:.3f}",
        va="center",
        color=COLORS["text"],
        fontweight="bold",
    )

save_figure(fig, "04_model_f1_leaderboard.png")


# Compare all metrics
metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
]

fig, ax = plt.subplots(
    figsize=(13, max(6, 0.52 * len(results_df)))
)

plot_df = (
    results_df.sort_values("F1 Score", ascending=True)
    .reset_index(drop=True)
)

y_positions = np.arange(len(plot_df))
bar_height = 0.19

for i, metric in enumerate(metric_columns):
    ax.barh(
        y_positions + (i - 1.5) * bar_height,
        plot_df[metric],
        height=bar_height,
        label=metric,
        color=PALETTE[i],
    )

ax.set_yticks(y_positions)
ax.set_yticklabels(plot_df["Model"])
ax.set_xlim(0, 1.0)

style_axis(
    ax,
    "All models | Metric comparison",
    "Score",
    "Model",
    grid_axis="x",
)

ax.grid(axis="y", visible=False)
ax.legend(ncol=4, loc="lower right", frameon=True)

save_figure(fig, "05_model_metrics_comparison.png")


# ============================================================
# 10. BEST MODEL EVALUATION
# ============================================================

best_name = results_df.iloc[0]["Model"]

print(f"\nBest validation model: {best_name}")

if best_name == "PyTorch Neural Network" and TORCH_AVAILABLE:
    torch_model.eval()

    with torch.no_grad():
        best_pred = (
            torch_model(
                torch.tensor(
                    X_valid_scaled,
                    dtype=torch.float32,
                ).to(DEVICE)
            )
            .argmax(dim=1)
            .cpu()
            .numpy()
        )
else:
    best_model, best_input_type = trained_models[best_name]

    best_x_valid = (
        X_valid_scaled
        if best_input_type == "scaled"
        else X_valid_processed
    )

    best_pred = best_model.predict(best_x_valid)

print("\nClassification report:")
print(
    classification_report(
        y_valid,
        best_pred,
        zero_division=0,
    )
)


# Confusion matrix
cm = confusion_matrix(
    y_valid,
    best_pred,
    labels=[0, 1],
)

fig, ax = plt.subplots(figsize=(6.5, 5.5))

image = ax.imshow(cm, cmap="magma")

ax.set_xticks(
    [0, 1],
    labels=["Predicted 0", "Predicted 1"],
)

ax.set_yticks(
    [0, 1],
    labels=["Actual 0", "Actual 1"],
)

style_axis(
    ax,
    f"Confusion matrix | {best_name}",
    "Prediction",
    "Actual",
    grid_axis="both",
)

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white",
            fontsize=15,
            fontweight="bold",
        )

fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

save_figure(fig, "06_best_model_confusion_matrix.png")


# Random Forest feature importance
if "Random Forest" in trained_models:
    rf_model = trained_models["Random Forest"][0]

    try:
        feature_names = preprocessor.get_feature_names_out()

        importances = pd.Series(
            rf_model.feature_importances_,
            index=feature_names,
        ).sort_values().tail(15)

        fig, ax = plt.subplots(figsize=(11, 7))

        ax.barh(
            importances.index,
            importances.values,
            color=COLORS["cyan"],
        )

        style_axis(
            ax,
            "Random Forest | Top feature importances",
            "Importance",
            "Feature",
            grid_axis="x",
        )

        ax.grid(axis="y", visible=False)

        save_figure(
            fig,
            "07_random_forest_feature_importance.png",
        )

    except Exception as exc:
        print("Feature importance chart skipped:", exc)


# ============================================================
# 11. RETRAIN THE WINNER AND CREATE KAGGLE SUBMISSION
# ============================================================

classical_results = results_df[
    results_df["Model"].isin(trained_models.keys())
]

use_torch = (
    best_name == "PyTorch Neural Network"
    and TORCH_AVAILABLE
)

if use_torch:
    # Refit preprocessing on all labeled data
    preprocessor.fit(X)

    X_full_processed = np.asarray(
        preprocessor.transform(X),
        dtype=np.float32,
    )

    X_test_final_processed = np.asarray(
        preprocessor.transform(X_test),
        dtype=np.float32,
    )

    scaler.fit(X_full_processed)

    X_full_scaled = scaler.transform(
        X_full_processed
    ).astype(np.float32)

    X_test_final_scaled = scaler.transform(
        X_test_final_processed
    ).astype(np.float32)

    final_model = TitanicMLP(
        X_full_scaled.shape[1]
    ).to(DEVICE)

    full_x = torch.tensor(
        X_full_scaled,
        dtype=torch.float32,
    )

    full_y = torch.tensor(
        y.to_numpy(),
        dtype=torch.long,
    )

    full_loader = DataLoader(
        TensorDataset(full_x, full_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    final_optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    final_criterion = nn.CrossEntropyLoss()

    for epoch in range(60):
        final_model.train()

        for xb, yb in full_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            final_optimizer.zero_grad()

            loss = final_criterion(
                final_model(xb),
                yb,
            )

            loss.backward()
            final_optimizer.step()

    final_model.eval()

    with torch.no_grad():
        test_pred = (
            final_model(
                torch.tensor(
                    X_test_final_scaled,
                    dtype=torch.float32,
                ).to(DEVICE)
            )
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

else:
    # Select the best classical model by validation F1
    chosen_name = classical_results.iloc[0]["Model"]

    chosen_model, chosen_input_type = trained_models[chosen_name]

    print(f"\nFinal submission model: {chosen_name}")

    # Refit preprocessing on all labeled data
    preprocessor.fit(X)

    X_full_processed = np.asarray(
        preprocessor.transform(X),
        dtype=np.float32,
    )

    X_test_final_processed = np.asarray(
        preprocessor.transform(X_test),
        dtype=np.float32,
    )

    if chosen_input_type == "scaled":
        scaler.fit(X_full_processed)

        X_full_final = scaler.transform(
            X_full_processed
        )

        X_test_final = scaler.transform(
            X_test_final_processed
        )
    else:
        X_full_final = X_full_processed
        X_test_final = X_test_final_processed

    chosen_model.fit(X_full_final, y)

    test_pred = chosen_model.predict(X_test_final)


# Create submission file
submission = pd.DataFrame({
    "PassengerId": test_raw["PassengerId"],
    "Survived": np.asarray(test_pred).astype(int),
})

submission_path = OUTPUT_DIR / "submission.csv"

submission.to_csv(
    submission_path,
    index=False,
)

print("\nSubmission saved to:", submission_path)
print("\nSubmission preview:")
print(submission.head().to_string(index=False))

print("\n" + "=" * 68)
print("PROJECT COMPLETED")
print("=" * 68)
print("Charts and CSV files are in:", OUTPUT_DIR)
