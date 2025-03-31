import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import winsound

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
drowsy_counter = 0
awake_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    
    for face in faces:
        shape = shape_predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        if ear < EYE_AR_THRESH:
            drowsy_counter += 1
            awake_counter = 0
            status = "DROWSY! ALERT!"
            color = (0, 0, 255)
            if drowsy_counter >= EYE_AR_CONSEC_FRAMES:
                winsound.Beep(1000, 500)
        else:
            awake_counter += 1
            if awake_counter >= 5:
                drowsy_counter = 0
            status = "AWAKE"
            color = (0, 255, 0)
        
        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, color, 2)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, color, 2)
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()