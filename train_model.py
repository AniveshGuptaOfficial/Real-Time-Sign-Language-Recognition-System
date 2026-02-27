import pandas as pd
from sklearn.svm import SVC
import pickle
import numpy as np

print("Loading dataset...")

data = pd.read_csv("dataset.csv", header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# -------------------------------
# BETTER NORMALIZATION
# Make landmarks relative to wrist (point 0)
# -------------------------------
X = X.reshape(-1, 21, 2)
X = X - X[:, 0:1, :]        # subtract wrist coordinates
X = X.reshape(-1, 42)

print("Training model...")

model = SVC(kernel='rbf', probability=True)  # rbf is better than linear

model.fit(X, y)

pickle.dump(model, open("sign_model.pkl", "wb"))

print("Model Trained and Saved Successfully!")