import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import pickle

# ---------------------------------------------------
# Load the trained machine learning model
# The model was trained using 42 landmark features
# (21 points × 2 coordinates)
# ---------------------------------------------------
model = pickle.load(open("sign_model.pkl", "rb"))

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

# Initialize MediaPipe modules
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Start webcam capture
cap = cv2.VideoCapture(0)

# Variables for sentence formation
sentence = ""          # Stores final constructed sentence
stable_prediction = "" # Stores stable detected letter
counter = 0            # Frame counter for stability check
threshold = 12         # Number of consistent frames required

# ---------------------------------------------------
# Create MediaPipe Hand Detection Object
# ---------------------------------------------------
with mp_hands.Hands(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:

        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR → RGB (MediaPipe requires RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        results = hands.process(image)

        # Convert back RGB → BGR for OpenCV display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ---------------------------------------------------
        # If hand is detected
        # ---------------------------------------------------
        if results.multi_hand_landmarks:
            h, w, c = image.shape  # for bounding box

            for hand_landmark in results.multi_hand_landmarks:

                # Draw hand skeleton on screen
                mp_draw.draw_landmarks(
                    image,
                    hand_landmark,
                    mp_hands.HAND_CONNECTIONS
                )

                lmList = []
                xs, ys = [], []

                # ---------------------------------------------------
                # Extract 21 landmark coordinates (x, y)
                # Total Features = 21 × 2 = 42
                # ---------------------------------------------------
                for lm in hand_landmark.landmark:
                    xs.append(lm.x)
                    ys.append(lm.y)
                    lmList.append([lm.x, lm.y])

                lmList = np.array(lmList)

                # ---------------------------------------------------
                # Normalize relative to wrist (landmark 0)
                # Makes system translation-invariant
                # ---------------------------------------------------
                lmList = lmList - lmList[0]

                # Flatten to 1D feature vector
                lmList = lmList.flatten()

                # ---------------------------------------------------
                # Predict using trained SVM model
                # ---------------------------------------------------
                probs = model.predict_proba([lmList])[0]
                max_prob = float(np.max(probs))
                prediction = model.classes_[np.argmax(probs)]

                # ---------------------------------------------------
                # Stability Check:
                # Accept letter only if detected for
                # multiple consecutive frames
                # ---------------------------------------------------
                counter += 1
                if counter > threshold:
                    stable_prediction = prediction
                    counter = 0

                # ---------------------------------------------------
                # Compute bounding box around hand
                # ---------------------------------------------------
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                x1, y1 = int(min_x * w), int(min_y * h)
                x2, y2 = int(max_x * w), int(max_y * h)

                # Expand the bounding box (larger box around hand)
                margin = 50  # increase/decrease to change box size
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)

                # Draw bounding box (rectangle around the hand)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # Prepare label text with confidence
                label_text = f"{stable_prediction} ({max_prob:.2f})"

                # Draw label just above the bounding box
                cv2.putText(
                    image,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2
                )

        # ---------------------------------------------------
        # Display detected letter (large text)
        # ---------------------------------------------------
        cv2.putText(
            image,
            f"Letter: {stable_prediction}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        # ---------------------------------------------------
        # Display constructed sentence
        # ---------------------------------------------------
        cv2.putText(
            image,
            f"Sentence: {sentence}",
            (10, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        # Show output window
        cv2.imshow("Sign Language Recognition", image)

        # Wait for key press
        key = cv2.waitKey(1)

        # ---------------------------------------------------
        # Keyboard Controls
        # ---------------------------------------------------

        # Add detected letter to sentence
        if key == ord('s') and stable_prediction != "":
            sentence += stable_prediction

        # Add SPACE
        if key == 32:  # ASCII 32 = Spacebar
            sentence += " "

        # BACKSPACE (delete last character)
        if key == 8:  # ASCII 8 = Backspace
            sentence = sentence[:-1]

        # Clear entire sentence
        if key == ord('c'):
            sentence = ""

        # Convert sentence to speech
        if key == ord('t'):
            engine.say(sentence)
            engine.runAndWait()

        # Quit program
        if key == ord('q'):
            break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
