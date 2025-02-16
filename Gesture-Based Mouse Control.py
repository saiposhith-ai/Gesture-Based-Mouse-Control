import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen size
screen_w, screen_h = pyautogui.size()
prev_x, prev_y = 0, 0  # To store previous cursor position
smoothing_factor = 5  # To smooth cursor movement

# Gesture thresholds
click_threshold = 0.02  # Distance threshold for clicking
scroll_threshold = 0.05  # Distance threshold for scrolling
right_click_threshold = 0.04  # Distance threshold for right-clicking
swipe_threshold = 0.1  # Threshold for three-finger swipe
click_cooldown = 0.5  # Cooldown time to prevent multiple rapid clicks
last_click_time = 0

# Capture Video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with Mediapipe
    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get index finger tip (8), middle finger tip (12), and thumb tip (4)
            index_x, index_y = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
            middle_x, middle_y = int(hand_landmarks.landmark[12].x * w), int(hand_landmarks.landmark[12].y * h)
            thumb_x, thumb_y = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
            
            # Convert index finger position to screen coordinates for cursor movement
            screen_x = np.interp(index_x, [0, w], [0, screen_w])
            screen_y = np.interp(index_y, [0, h], [0, screen_h])
            
            # Smooth cursor movement
            cur_x = prev_x + (screen_x - prev_x) / smoothing_factor
            cur_y = prev_y + (screen_y - prev_y) / smoothing_factor
            pyautogui.moveTo(cur_x, cur_y)
            prev_x, prev_y = cur_x, cur_y
            
            # Scroll Gesture (Middle and Index fingers together moving up/down)
            scroll_distance = abs(index_y - middle_y) / h
            if scroll_distance < scroll_threshold:
                if index_y < middle_y:
                    pyautogui.scroll(10)
                else:
                    pyautogui.scroll(-10)
            
            # Click Gesture (Pinching Thumb and Index together with cooldown)
            pinch_distance = np.linalg.norm(np.array([index_x, index_y]) - np.array([thumb_x, thumb_y])) / w
            current_time = time.time()
            if pinch_distance < click_threshold and (current_time - last_click_time > click_cooldown):
                pyautogui.click()
                last_click_time = current_time  # Update last click time

    # Show output
    cv2.imshow('Hand Tracking Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
