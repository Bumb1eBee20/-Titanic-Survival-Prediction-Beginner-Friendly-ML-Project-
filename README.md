# 🛳️ Titanic Survival Prediction – Machine Learning Project

> **"Would you have survived the Titanic?"**  
This project predicts survival using classic machine learning pipelines — built from scratch by a beginner, for beginners.

This is a complete end-to-end **data science project** based on the Titanic dataset. It includes EDA, data preprocessing, logistic regression model training, evaluation, and saving. The project is organized like a real-world ML repo, perfect for portfolios and showcasing to recruiters.

---

## 📌 Highlights

- ✅ Clean project structure
- 🧠 Beginner-friendly ML pipeline
- 🔎 Full EDA with visualizations
- 💾 Model saved using `pickle`
- 📂 Modular Python scripts + notebook
- ⚙️ Ready for upgrades (Streamlit, APIs, other models)
---
## 🧠 Tech Stack

![Status](https://img.shields.io/badge/status-active-success?style=flat-square)
- **Language**: Python  ![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `pickle`
- **Model Used**: Logistic Regression  ![ML](https://img.shields.io/badge/Machine_Learning-logistic_regression-orange?style=flat-square)
- **Type**: Supervised Binary Classification
- **License** : ![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
---

## 📁 Folder Structure

```

titanic-survival-dashboard/
├── data/
│   └── titanic.csv                 # Dataset
├── models/
│   └── model.pkl                   # Trained model
├── notebooks/
│   └── eda\_notebook.ipynb         # Full EDA and data visualizations
├── src/
│   ├── data\_preprocessing.py      # Null handling, encoding, splitting
│   └── model.py                   # Model training, evaluation, saving
├── requirements.txt               # Python dependencies
├── .gitignore                     # Files to ignore
└── README.md                      # You're reading it!

````

---

## 🚀 How to Run This Project

1. **Install dependencies**

   ```bash
   git clone https://github.com/your-username/titanic-survival-dashboard.git
   cd titanic-survival-dashboard
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the model training**

   ```bash
   python src/model.py
   ```

4. ✅ Model is trained and saved in

   ```bash
   models/model.pkl
   ```
---

## 📊 Sample Output

```
✅ Model training complete!
🔍 Accuracy: 0.8231

📊 Classification Report:
              precision    recall  f1-score   support
         0       0.83      0.86      0.85       100
         1       0.81      0.77      0.79        56
    accuracy                           0.82       156
```

---

## ✨ Key Learnings

* How to handle missing values with `.fillna()`
* Encoding categorical features using `.map()`
* Why to split code into `.py` and `.ipynb`
* How to train and evaluate ML models
* Saving a model with `pickle` for reuse

---

## 💡 Next Steps (Future Upgrades)

* [ ] Add a `predict.py` CLI tool for single predictions
* [ ] Build a Streamlit UI for user input and survival prediction
* [ ] Try other ML models like Decision Tree, Random Forest, XGBoost
* [ ] Use `GridSearchCV` for hyperparameter tuning
* [ ] Deploy to web (Streamlit Cloud / Render / Hugging Face Spaces)

---

## 🙌 Author

**Bachupati Dimpu Narasimha Sai**
Final Year CSE Student | ML, Data Science & AI Enthusiast
💻 Passionate about real-world machine learning projects and deployment


## 🏁 Project Status

✅ Functional
🧪 Tested
📦 Structured
✨ Ready for upgrade 🚀

---
