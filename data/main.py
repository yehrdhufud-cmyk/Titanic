import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print("Training data:", train.shape)
print("Testing data:", test.shape)

print(train.head())

print(train["Survived"].value_counts())