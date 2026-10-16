import pandas as pd

train = pd.read_csv("data/train.csv")

print("Training data:", train.shape)

print(train.head())

print(train["Survived"].value_counts())


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

X = train[features]
y = train["Survived"]

print("Features:", X.shape)
print("Target:", y.shape)

print(X.isnull().sum())