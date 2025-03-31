import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks):
    """
    Count the number of extended fingers based on landmark positions.
    """
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    finger_dips = [3, 6, 10, 14, 18]  # Joints below the tips
    
    fingers = []

    # Thumb (special case: checks x-coordinates for left/right hand)
    if hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_dips[0]].x:
        fingers.append(1)  # Extended
    else:
        fingers.append(0)  # Folded

    # Other four fingers (compare Y-coordinates: tip should be above DIP)
    for tip, dip in zip(finger_tips[1:], finger_dips[1:]):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y:
            fingers.append(1)  # Extended
        else:
            fingers.append(0)  # Folded

    return sum(fingers)  # Total count of extended fingers

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Count fingers
            num_fingers = count_fingers(hand_landmarks)

            # Display the count
            cv2.putText(frame, f'Fingers: {num_fingers}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Finger Counter", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
