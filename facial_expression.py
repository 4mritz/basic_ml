import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break  # Exit if frame not captured

    try:
        # Analyze facial expression
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Get the detected emotion
        emotion = analysis[0]['dominant_emotion']

        # Display the emotion on the video feed
        cv2.putText(frame, f'Emotion: {emotion}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print(f"Error: {e}")

    # Show the live video feed
    cv2.imshow("Facial Expression Recognition", frame)

    # Add a short delay to allow the window to open
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
