import cv2
import mediapipe as mp
import numpy as np
import os
import csv

mp_hands = mp.solutions.hands

dataset_path = r"C:\Users\Anivesh\OneDrive\Desktop\Coding\Python\Gesture-Controlled-Automation-for-Hill-Climb-Racing-main\asl_dataset"

output_csv = "dataset.csv"

hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=1,
                       min_detection_confidence=0.5)

with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)

    for label in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, label)

        if not os.path.isdir(folder_path):
            continue

        print(f"Processing {label}...")

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)

            image = cv2.imread(image_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    lmList = []

                    for lm in hand_landmarks.landmark:
                        lmList.append([lm.x, lm.y])

                    lmList = np.array(lmList)

                    # Normalize relative to wrist
                    lmList = lmList - lmList[0]
                    lmList = lmList.flatten()

                    writer.writerow(list(lmList) + [label])

print("Dataset CSV Created Successfully!")