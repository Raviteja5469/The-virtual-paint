# module 1

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,    # Increased confidence threshold
    min_tracking_confidence=0.7      # Added tracking confidence
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Drawing variables
drawing_points = []
current_color = (0, 0, 255)  # Red color (BGR format)
thickness = 2
prev_x, prev_y = None, None

def calculate_distance(point1, point2):
    """Calculate distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def smooth_points(points, smoothing_factor=5):
    """Smooth points using moving average"""
    if len(points) < smoothing_factor:
        return points
    
    smoothed = []
    for i in range(len(points)):
        start_idx = max(0, i - smoothing_factor)
        end_idx = min(len(points), i + smoothing_factor + 1)
        window = points[start_idx:end_idx]
        smoothed.append((
            int(sum(p[0] for p in window) / len(window)),
            int(sum(p[1] for p in window) / len(window))
        ))
    return smoothed

while True:
    success, frame = cap.read()
    if not success:
        break
        
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hand landmarks
    results = hands.process(rgb_frame)
    
    # Create a drawing overlay
    overlay = frame.copy()
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
            )
            
            # Get finger landmarks
            index_tip = hand_landmarks.landmark[8]    # Index finger tip
            index_pip = hand_landmarks.landmark[6]    # Index finger PIP joint
            middle_tip = hand_landmarks.landmark[12]  # Middle finger tip
            thumb_tip = hand_landmarks.landmark[4]    # Thumb tip
            
            # Convert coordinates
            h, w, _ = frame.shape
            x, y = int(index_tip.x * w), int(index_tip.y * h)
            
            # Check if index finger is raised and middle finger is down
            if (index_tip.y < index_pip.y and     # Index finger is raised
                middle_tip.y > index_pip.y):      # Middle finger is down
                
                if prev_x is None:  # First point
                    prev_x, prev_y = x, y
                else:
                    # Calculate distance between current and previous point
                    distance = calculate_distance((x, y), (prev_x, prev_y))
                    
                    # Only draw if the movement is not too large (reduces jitter)
                    if distance < 50:  # Adjust this threshold as needed
                        drawing_points.append((x, y))
                        # Draw line
                        if len(drawing_points) > 1:
                            smoothed_points = smooth_points(drawing_points)
                            for i in range(1, len(smoothed_points)):
                                cv2.line(overlay, 
                                       smoothed_points[i-1], 
                                       smoothed_points[i], 
                                       current_color, 
                                       thickness)
                    prev_x, prev_y = x, y
            else:
                # Reset points when finger is down
                prev_x, prev_y = None, None
                if len(drawing_points) > 0:
                    # Save the current stroke
                    frame = overlay.copy()
                    drawing_points = []
    
    # Blend the overlay with the original frame
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Display the frame
    cv2.imshow('Hand Drawing', frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        current_color = (0, 0, 255)  # Red
    elif key == ord('g'):
        current_color = (0, 255, 0)  # Green
    elif key == ord('b'):
        current_color = (255, 0, 0)  # Blue
    elif key == ord('+'):
        thickness = min(thickness + 1, 10)
    elif key == ord('-'):
        thickness = max(thickness - 1, 1)
    elif key == ord('c'):
        # Clear drawing
        drawing_points = []
        frame = cv2.VideoCapture(0).read()[1]
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
