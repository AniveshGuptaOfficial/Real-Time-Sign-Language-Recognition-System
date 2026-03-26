import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# ---------------------------------------------------
# Initialize MediaPipe Hands Module
# ---------------------------------------------------
mp_hands = mp.solutions.hands

# ---------------------------------------------------
# Path to ASL image dataset folder
# Each subfolder name represents a class label
# Example:
# asl_dataset/
# ├── A/
# ├── B/
# ├── 0/
# ├── 1/
# ---------------------------------------------------
dataset_path = r"C:\Users\Anivesh\OneDrive\Desktop\Coding\Python\Real-Time-Sign-Language-Recognition-System\asl_dataset"

# Output CSV file where extracted features will be saved
output_csv = "dataset.csv"

# ---------------------------------------------------
# Create MediaPipe Hands Object
# static_image_mode=True → Because we are processing images (not video)
# max_num_hands=1 → Only one hand expected per image
# ---------------------------------------------------
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ---------------------------------------------------
# Open CSV file in write mode
# This will add new data without overwriting existing data
# ---------------------------------------------------
with open(output_csv, mode="a", newline="") as f:
    writer = csv.writer(f)

    # ---------------------------------------------------
    # Loop through each folder (label)
    # Each folder name is the class (A-Z, 0-9)
    # ---------------------------------------------------
    for label in os.listdir(dataset_path):

        folder_path = os.path.join(dataset_path, label)

        # Skip if not a folder
        if not os.path.isdir(folder_path):
            continue

        print(f"Processing {label}...")

        # ---------------------------------------------------
        # Loop through all images inside that label folder
        # ---------------------------------------------------
        for image_name in os.listdir(folder_path):

            image_path = os.path.join(folder_path, image_name)

            # Read image
            image = cv2.imread(image_path)

            # Skip invalid images
            if image is None:
                continue

            # Convert BGR → RGB (required for MediaPipe)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect hand landmarks in the image
            results = hands.process(image_rgb)

            # ---------------------------------------------------
            # If hand landmarks are detected
            # ---------------------------------------------------
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    lmList = []

                    # ---------------------------------------------------
                    # Extract 21 landmark points
                    # Each landmark contains:
                    # x, y, z (we use only x and y)
                    # ---------------------------------------------------
                    for lm in hand_landmarks.landmark:
                        lmList.append([lm.x, lm.y])

                    lmList = np.array(lmList)

                    # ---------------------------------------------------
                    # Normalize landmarks relative to wrist (landmark 0)
                    # This makes gesture position independent
                    # ---------------------------------------------------
                    lmList = lmList - lmList[0]

                    # Flatten into 1D array
                    # 21 landmarks × 2 coordinates = 42 features
                    lmList = lmList.flatten()

                    # ---------------------------------------------------
                    # Write feature vector + label to CSV
                    # Format:
                    # x1,y1,x2,y2,...,x21,y21,label
                    # ---------------------------------------------------
                    writer.writerow(list(lmList) + [label])

print("Dataset CSV Created Successfully!")