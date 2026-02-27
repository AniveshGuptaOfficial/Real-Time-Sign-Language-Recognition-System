import pandas as pd
from sklearn.svm import SVC
import pickle
import numpy as np

# ---------------------------------------------------
# Step 1: Load the Dataset
# dataset.csv contains:
# 42 feature columns (21 landmarks × 2 coordinates)
# + 1 label column (A-Z, 0-9, SPACE)
# ---------------------------------------------------
print("Loading dataset...")

data = pd.read_csv("dataset.csv", header=None)

# ---------------------------------------------------
# Step 2: Separate Features (X) and Labels (y)
# X → 42 numerical landmark features
# y → Class labels
# ---------------------------------------------------
X = data.iloc[:, :-1].values   # All columns except last
y = data.iloc[:, -1].values    # Last column (label)

# ---------------------------------------------------
# Step 3: Landmark Normalization
# Convert absolute coordinates into relative coordinates
# by subtracting wrist (landmark 0)
#
# Why?
# → Makes model invariant to hand position
# → Improves generalization
# ---------------------------------------------------

# Reshape into (samples, 21 landmarks, 2 coordinates)
X = X.reshape(-1, 21, 2)

# Subtract wrist coordinates from all landmarks
X = X - X[:, 0:1, :]

# Flatten back into (samples, 42 features)
X = X.reshape(-1, 42)

# ---------------------------------------------------
# Step 4: Train Support Vector Machine (SVM)
#
# Kernel: 'rbf'
# → Radial Basis Function
# → Better for non-linear gesture patterns
#
# probability=True
# → Enables predict_proba() for confidence scores
# ---------------------------------------------------
print("Training model...")

model = SVC(kernel='rbf', probability=True)

# Train the classifier
model.fit(X, y)

# ---------------------------------------------------
# Step 5: Save Trained Model
# The model is saved as sign_model.pkl
# ---------------------------------------------------
pickle.dump(model, open("sign_model.pkl", "wb"))

print("Model Trained and Saved Successfully!")