import streamlit as st
import pickle
import numpy as np

st.title("Titanic Survival Prediction")
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
pclass= st.selectbox("Passenger Class", [1, 2, 3])
sex= st.radio("Sex", ["Male", "Female"])
age= st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses", 0, 10)
parch= st.number_input("Parents/Children", 0, 10)
fare = st.number_input("Fare", 0.0, 500.0)
embarked =st.selectbox("Embarked", ["S", "C", "Q"])
sex = 0 if sex == "Male" else 1
embarked ={"S": 0, "C": 1, "Q": 2}[embarked]
features= np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
if st.button("Predict"):
    result= model.predict(features)[0]
    st.success("Survived" if result == 1 else "Did Not Survive")
