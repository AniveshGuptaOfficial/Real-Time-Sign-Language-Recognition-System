import cv2
import mediapipe as mp
import numpy as np
import csv

# ---------------------------------------------------
# Initialize MediaPipe Hand Detection Module
# ---------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---------------------------------------------------
# Start Webcam Capture
# ---------------------------------------------------
cap = cv2.VideoCapture(0)

print("Press:")
print("A-Z → Collect alphabet")
print("0-9 → Collect digit")
print("SPACEBAR → Collect SPACE")
print("Q → Quit")

# Variable to store current class label
current_label = None

# Counter to track number of samples collected
sample_count = 0

# ---------------------------------------------------
# Create MediaPipe Hand Object
# ---------------------------------------------------
with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1        # Detect only one hand
) as hands:

    while True:

        # Capture frame from webcam
        ret, frame = cap.read()

        if not ret:
            print("Camera not working")
            break

        # Convert BGR to RGB (MediaPipe requires RGB)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process image to detect hand landmarks
        results = hands.process(image_rgb)

        # Convert back to BGR for OpenCV display
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # ---------------------------------------------------
        # If hand is detected
        # ---------------------------------------------------
        if results.multi_hand_landmarks:

            print("Hand Detected")

            for hand_landmark in results.multi_hand_landmarks:

                # Draw hand landmarks on image
                mp_draw.draw_landmarks(
                    image,
                    hand_landmark,
                    mp_hands.HAND_CONNECTIONS
                )

                # ---------------------------------------------------
                # Extract 21 landmark coordinates (x, y)
                # Total features = 21 × 2 = 42
                # ---------------------------------------------------
                lmList = []

                for lm in hand_landmark.landmark:
                    lmList.append(lm.x)   # x coordinate
                    lmList.append(lm.y)   # y coordinate

                # ---------------------------------------------------
                # Save data only if a label is selected
                # ---------------------------------------------------
                if current_label is not None:

                    with open("dataset.csv", "a", newline="") as f:
                        writer = csv.writer(f)

                        # Save 42 features + label
                        writer.writerow(lmList + [current_label])

                    sample_count += 1
                    print(f"Saved: {current_label} | Total: {sample_count}")

        # ---------------------------------------------------
        # Display Current Label & Sample Count on Screen
        # ---------------------------------------------------
        cv2.putText(image,
                    f"Current Label: {current_label}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.putText(image,
                    f"Samples: {sample_count}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)

        # Show video window
        cv2.imshow("Collecting Data", image)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # ---------------------------------------------------
        # Key Controls for Label Selection
        # ---------------------------------------------------

        # Quit program
        if key == ord('q'):
            break

        # SPACEBAR → Label as "SPACE"
        elif key == 32:
            current_label = "SPACE"
            sample_count = 0

        # Digits 0–9
        elif 48 <= key <= 57:
            current_label = chr(key)
            sample_count = 0

        # Lowercase letters → Convert to uppercase
        elif 97 <= key <= 122:
            current_label = chr(key).upper()
            sample_count = 0

        # Uppercase letters A–Z
        elif 65 <= key <= 90:
            current_label = chr(key)
            sample_count = 0

# Release camera and close all windows
cap.release()
cv2.destroyAllWindows()