# module 2

import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Initialize Camera
cap = cv2.VideoCapture(0)


def detect_hand_landmarks(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    return results.multi_hand_landmarks


def draw_on_canvas(landmarks, frame, canvas, prev_point):
    # Get index finger tip coordinates (landmark 8)
    h, w, _ = frame.shape
    index_finger = landmarks.landmark[8]
    x, y = int(index_finger.x * w), int(index_finger.y * h)
    
    if prev_point is None:
        prev_point = (x, y)
    
    # Draw line between previous and current point
    cv2.line(canvas, prev_point, (x, y), (255, 255, 255), 2)
    return (x, y)


prev_point = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)  # Mirror the frame
    hand_landmarks = detect_hand_landmarks(frame)
    
    if hand_landmarks:
        for landmarks in hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Update drawing
            prev_point = draw_on_canvas(landmarks, frame, canvas, prev_point)
    else:
        prev_point = None
    
    # Show both frame and canvas
    cv2.imshow('Hand Tracking', frame)
    cv2.imshow('Drawing Canvas', canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Add color selection based on hand gesture
def get_drawing_color(landmarks):
    # Example: Use thumb position relative to index finger
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]
    
    if thumb_tip.y < index_tip.y:
        return (0, 0, 255)  # Red
    else:
        return (255, 255, 255)  # White
