import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from gtts import gTTS
import pygame
import os

# Load your hand gesture model dictionary
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define your gesture label dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello',
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please',
    32: 'You are welcome.'
}

# Initialize variables
current_word = ""
previous_character = None

# Initialize time for gesture
waktu_gestur = time.time()

# Initialize pygame for audio playback
pygame.mixer.init()

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

# Load the hat image with an alpha channel (transparency)
hat_ori = cv2.imread('coba.png', -1)
glass_ori = cv2.imread('pink.png', -1)

# Define the desired size for the hat and glasses images (enlarged by a factor of 3)
new_width = int(hat_ori.shape[1] * 3)
new_height = int(hat_ori.shape[0] * 3)

# Resize the hat and glasses images
hat_ori_resized = cv2.resize(hat_ori, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
glass_ori_resized = cv2.resize(glass_ori, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

    try:
        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]
        print("Predicted character : ", predicted_character)

        if predicted_character == 'I Love you':
            # Augmented Reality Code
            faces = face_cascade.detectMultiScale(frame, 1.2, 5, 0, (120, 120), (350, 350))
            for (x, y, w, h) in faces:
                # Adjust the position and size of the hat and glasses based on hand landmarks
                hat_symin = int(y - 8 * h / 24)
                hat_symax = int(y - 1 * h / 24)
                sh_hat = hat_symax - hat_symin

                glass_symin = int(y + 1.5 * h / 4)
                glass_symax = int(y + 2.5 * h / 4)
                sh_glass = glass_symax - glass_symin

                face_glass_roi_color = frame[glass_symin:glass_symax, x:x + w]
                hat_roi_color = frame[hat_symin:hat_symax, x:x + w]

                glass = cv2.resize(glass_ori_resized, (w, sh_glass), interpolation=cv2.INTER_CUBIC)
                hat = cv2.resize(hat_ori_resized, (w, sh_hat), interpolation=cv2.INTER_CUBIC)

                def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
                    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
                    h, w, _ = overlay.shape
                    rows, cols, _ = src.shape
                    y, x = pos[0], pos[1]

                    for i in range(h):
                        for j in range(w):
                            if x + i >= rows or y + j >= cols:
                                continue
                            alpha = float(overlay[i][j][3] / 255.0)
                            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
                    return src

                transparentOverlay(face_glass_roi_color, glass)
                transparentOverlay(hat_roi_color, hat)

        if time.time() - waktu_gestur >= 3:
            if predicted_character != previous_character:
                current_word += predicted_character
                previous_character = predicted_character

            waktu_gestur = time.time()

        cv2.putText(frame, current_word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    except Exception as e:
        pass

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        print("Kalimat Akhir:", current_word)

        text_to_speak = "Kalimat Akhir: " + current_word
        tts = gTTS(text=text_to_speak, lang='id')
        tts.save("output_audio.mp3")

        pygame.mixer.music.load("output_audio.mp3")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        break

pygame.mixer.quit()
cap.release()
cv2.destroyAllWindows()
