import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import pickle

# Load trained model
model = pickle.load(open("sign_model.pkl", "rb"))
engine = pyttsx3.init()

mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

sentence = ""
stable_prediction = ""
counter = 0
threshold = 12   # number of stable frames required

with mp_hands.Hands(min_detection_confidence=0.6,
                    min_tracking_confidence=0.6) as hands:

    while True:
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(image,
                                       hand_landmark,
                                       mp_hands.HAND_CONNECTIONS)

                lmList = []

                for lm in hand_landmark.landmark:
                    lmList.append([lm.x, lm.y])

                lmList = np.array(lmList)

                # -------------------------------
                # SAME NORMALIZATION AS TRAINING
                # -------------------------------
                lmList = lmList - lmList[0]  # subtract wrist
                lmList = lmList.flatten()

                probs = model.predict_proba([lmList])[0]
                max_prob = np.max(probs)
                prediction = model.classes_[np.argmax(probs)]

                # Accept only confident predictions
                if True:
                    counter += 1
                    if counter > threshold:
                        stable_prediction = prediction
                        counter = 0
                else:
                    counter = 0

                cv2.putText(image, f"Letter: {stable_prediction}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 255, 0), 3)

        cv2.putText(image, f"Sentence: {sentence}",
                    (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)

        cv2.imshow("Sign Language Recognition", image)

        key = cv2.waitKey(1)

        # Add detected letter
        if key == ord('s') and stable_prediction != "":
            sentence += stable_prediction

        # Add SPACE
        if key == 32:   # Spacebar
            sentence += " "

        # BACKSPACE (delete last character)
        if key == 8:   # Backspace key
            sentence = sentence[:-1]

        # Clear entire sentence
        if key == ord('c'):
            sentence = ""

        # Text to speech
        if key == ord('t'):
            engine.say(sentence)
            engine.runAndWait()

        # Quit
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()