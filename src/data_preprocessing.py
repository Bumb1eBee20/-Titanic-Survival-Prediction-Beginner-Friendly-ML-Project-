import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data():
    df = pd.read_csv("data/titanic.csv")
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = df.drop(columns=["Cabin"])
    df = df.drop_duplicates()

    # Encoding
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    # Feature Selection
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df[features]
    y = df["Survived"]

    return train_test_split(X, y, test_size=0.2, random_state=42)
