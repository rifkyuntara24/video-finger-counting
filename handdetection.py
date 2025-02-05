import cv2
import mediapipe as mp
import pyttsx3
import time

engine = pyttsx3.init()
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

# Varibel untuk menunda suara yang keluar
last_spoken = 0
cooldown = 2  # seconds

def count_fingers(handLms):
    finger_tips = [4, 8, 12, 16, 20]  # merepresentasikan tip (puncak) ibu jari, telunjuk, tengah, manis, kelingking 
    finger_states = [0, 0, 0, 0, 0]
    
    # Perbadingan Antara Number setiap landmark pada posisi di frame kamera jika frame lebih kecil (jari terangkat)
    # Cek posisi Ibu jari menggunakkan X, karena posisi ibu jari horzontal
    if handLms.landmark[4].x < handLms.landmark[3].x:
        finger_states[0] = 1

    # Cek jari lainnya menggunakan Y, karena posisi jari vertikal
    for i in range(1, 5):
        tip_y = handLms.landmark[finger_tips[i]].y
        pip_y = handLms.landmark[finger_tips[i]-2].y
        if tip_y < pip_y:
            finger_states[i] = 1

    return sum(finger_states)


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            count = count_fingers(handLms)
            # Menampilkan Jumlah jari pada frame
            cv2.putText(img, f"Fingers: {count}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # Memunculkan suara dengan waktu cooldown
            if time.time() - last_spoken > cooldown:
                engine.say(f"There are {count} finger")
                engine.runAndWait()
                last_spoken = time.time()
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                # if id == 4:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 300), cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 255, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)