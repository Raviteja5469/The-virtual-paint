# module 2    a bit accurate working module

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Get frame dimensions
_, frame = cap.read()
height, width = frame.shape[:2]

# Color options
colors = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'black': (0, 0, 0),
    'pink': (147, 20, 255)
}

# Color buttons
button_width = width // len(colors)
button_height = 50
buttons = {}
x = 0
for color_name, color_value in colors.items():
    buttons[color_name] = {
        'rect': (x, 0, button_width, button_height),
        'color': color_value
    }
    x += button_width

# Drawing variables
current_color = colors['black']
thickness = 3
drawing_points = []
canvas = np.zeros((height, width, 3), dtype=np.uint8)
prev_x, prev_y = None, None
eraser_mode = False

def is_point_in_rect(point, rect):
    """Check if point is inside rectangle"""
    x, y = point
    rx, ry, rw, rh = rect
    return rx < x < rx + rw and ry < y < ry + rh

def draw_color_palette(frame):
    """Draw color selection buttons"""
    for color_name, button in buttons.items():
        x, y, w, h = button['rect']
        color = button['color']
        cv2.rectangle(frame, (x, y), (x + w, h), color, -1)
        # Add color name text
        text_size = cv2.getTextSize(color_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(frame, color_name, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

while True:
    success, frame = cap.read()
    if not success:
        break
        
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Add the canvas to the frame
    mask = canvas.astype(bool).any(axis=2)
    frame[mask] = canvas[mask]
    
    # Draw color palette
    draw_color_palette(frame)
    
    # Process hand landmarks
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            
            # Get index finger tip coordinates
            index_tip = hand_landmarks.landmark[8]
            index_pip = hand_landmarks.landmark[6]
            middle_tip = hand_landmarks.landmark[12]
            
            x, y = int(index_tip.x * width), int(index_tip.y * height)
            
            # Check for color selection
            if y < button_height:
                for color_name, button in buttons.items():
                    if is_point_in_rect((x, y), button['rect']):
                        if color_name == 'black':  # Use black as eraser
                            eraser_mode = True
                            current_color = (0, 0, 0)
                        else:
                            eraser_mode = False
                            current_color = button['color']
            
            # Drawing logic
            if index_tip.y < index_pip.y and middle_tip.y > index_pip.y:  # Index finger raised, middle finger down
                if prev_x is None:
                    prev_x, prev_y = x, y
                else:
                    if eraser_mode:
                        # Erase in a circular area
                        cv2.circle(canvas, (x, y), 20, (0, 0, 0), -1)
                    else:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, thickness)
                    prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None
    
    # Display the frame
    cv2.imshow('Hand Drawing', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
