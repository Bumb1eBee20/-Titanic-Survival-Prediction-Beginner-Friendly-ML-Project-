import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import preprocess_data

MODEL_PATH = "models/model.pkl"

def train_model():
    if os.path.exists(MODEL_PATH):
        print(" model.pkl already exists. Skipping training.")
        return

    print(" Training model...")
    X_train, X_test, y_train, y_test = preprocess_data()
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f" Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\n Classification Report:\n", classification_report(y_test, y_pred))
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f" Saved model at: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
