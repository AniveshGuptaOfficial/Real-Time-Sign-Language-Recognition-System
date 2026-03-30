import cv2
import mediapipe as mp
import numpy as np
import pickle
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# -----------------------------
# Load trained model
# -----------------------------
model = pickle.load(open(r"C:\Users\Anivesh\OneDrive\Desktop\Coding\Python\RTSLRS\sign_model.pkl", "rb"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,   # relaxed so detection is easier
    min_tracking_confidence=0.3,
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

sentence = ""  # global sentence


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # read image from browser
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # mediapipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    print("multi_hand_landmarks:", results.multi_hand_landmarks)  # DEBUG

    if not results.multi_hand_landmarks:
        return {"letter": "", "confidence": 0.0}

    hand_landmark = results.multi_hand_landmarks[0]

    lmList = []
    for lm in hand_landmark.landmark:
        lmList.append([lm.x, lm.y])

    lmList = np.array(lmList)
    lmList = lmList - lmList[0]          # normalize to wrist
    features = lmList.flatten().reshape(1, -1)

    probs = model.predict_proba(features)[0]
    max_prob = float(np.max(probs))
    prediction = str(model.classes_[np.argmax(probs)])

    return {"letter": prediction, "confidence": round(max_prob, 3)}


@app.post("/sentence")
async def sentence_op(action: str = Form(...), letter: str = Form("")):
    global sentence

    if action == "add" and letter:
        sentence += letter
    elif action == "space":
        sentence += " "
    elif action == "backspace":
        sentence = sentence[:-1]
    elif action == "clear":
        sentence = ""

    return {"sentence": sentence}