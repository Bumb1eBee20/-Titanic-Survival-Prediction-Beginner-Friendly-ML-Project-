from data_preprocessing import preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

X_train, X_test, y_train, y_test = preprocess_data()
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Model training complete!")
print(f"Accuracy: {acc:.4f}")
print("\n Classification Report:\n", classification_report(y_test, y_pred))

# Step 4: Save model
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print(" Model saved to 'models/model.pkl'")
