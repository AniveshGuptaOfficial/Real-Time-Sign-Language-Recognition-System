import cv2
import mediapipe as mp
import numpy as np
import csv

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Press:")
print("A-Z → Collect alphabet")
print("0-9 → Collect digit")
print("SPACEBAR → Collect SPACE")
print("Q → Quit")

current_label = None
sample_count = 0

with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1
) as hands:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Camera not working")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # If hand detected
        if results.multi_hand_landmarks:
            print("Hand Detected")

            for hand_landmark in results.multi_hand_landmarks:

                # Draw landmarks
                mp_draw.draw_landmarks(
                    image,
                    hand_landmark,
                    mp_hands.HAND_CONNECTIONS
                )

                lmList = []

                for lm in hand_landmark.landmark:
                    lmList.append(lm.x)
                    lmList.append(lm.y)

                if current_label is not None:
                    with open("dataset.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(lmList + [current_label])

                    sample_count += 1
                    print(f"Saved: {current_label} | Total: {sample_count}")

        # Display current label
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

        cv2.imshow("Collecting Data", image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == 32:  # Spacebar
            current_label = "SPACE"
            sample_count = 0

        elif 48 <= key <= 57:  # Digits 0-9
            current_label = chr(key)
            sample_count = 0

        elif 97 <= key <= 122:  # lowercase letters
            current_label = chr(key).upper()
            sample_count = 0

        elif 65 <= key <= 90:  # uppercase letters
            current_label = chr(key)
            sample_count = 0

cap.release()
cv2.destroyAllWindows()