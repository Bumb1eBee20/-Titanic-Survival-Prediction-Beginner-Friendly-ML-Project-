import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def preprocess_data(path="Data/Titanic-Dataset.csv"):
    df=pd.read_csv(path)
    # Removing and Filling the null Values
    df["Age"]=df["Age"].fillna(df["Age"].mean())
    df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])
    df=df.drop(columns=["Cabin","Name","Ticket","PassengerId"])
    df=df.drop_duplicates()

    df["Sex"]=df["Sex"].map({"male": 0,"female": 1})
    df["Embarked"]=df["Embarked"].map({"S": 0,"C": 1,"Q": 2})

    features=["Age","Sex","Embarked","Pclass","SibSp","Parch","Fare"]
    X=df[features]
    y=df["Survived"]
    return train_test_split(X,y,test_size=0.2,random_state=42)
if __name__=="__main__":
    X_train,X_test,y_train,y_test=preprocess_data()
    print("Data Preprocessing Complete!!!")
    print(f"Train shape : {X_train.shape}\n Test shape : {X_test.shape}\n")
    print(f'Sample X:\n{X_train.head()}')