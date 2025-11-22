import cv2 as cv
import mediapipe as mp 
import pyautogui

x1 = y1 = x2 = y2 = 0

# Initialize webcam
webcam = cv.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize MediaPipe Hands
my_hands = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

while True:
    ret, image = webcam.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Flip the image for mirror effect
    image = cv.flip(image, 1)
    frame_height, frame_width, _ = image.shape
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(image, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Index fingertip
                    cv.circle(img=image, center=(x, y), radius=8, color=(0, 255, 255), thickness=3)
                    x1 = x
                    y1 = y

                if id == 4:  # Thumb tip
                    cv.circle(img=image, center=(x, y), radius=8, color=(0, 0, 255), thickness=3)
                    x2 = x
                    y2 = y

        # Calculate distance between index fingertip and thumb tip
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 // 4
        cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # Volume control based on distance
        if distance > 50:
            pyautogui.press('volumeup')
        else:
            pyautogui.press('volumedown')

    # Show frame
    cv.imshow('Hand Volume Control', image)

    # Exit on ESC key
    if cv.waitKey(1) & 0xFF == 27:
        break

# Release resources
webcam.release()
cv.destroyAllWindows()
