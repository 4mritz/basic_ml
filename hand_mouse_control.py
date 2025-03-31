import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture Video
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    frame_height, frame_width, _ = frame.shape  # Get frame dimensions

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x = int(index_finger.x * frame_width)
            y = int(index_finger.y * frame_height)

            # Scale coordinates to match screen resolution
            screen_x = np.interp(x, [0, frame_width], [0, screen_width])
            screen_y = np.interp(y, [0, frame_height], [0, screen_height])

            # Move mouse smoothly
            pyautogui.moveTo(screen_x, screen_y, duration=0.1)

            # Get thumb tip
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_x, thumb_y = int(thumb.x * frame_width), int(thumb.y * frame_height)

            # Calculate distance between index and thumb
            distance = np.linalg.norm(np.array([x, y]) - np.array([thumb_x, thumb_y]))

            # Click when fingers are close
            if distance < 20:  # Adjust sensitivity
                pyautogui.click()

    cv2.imshow("Hand Tracking Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
