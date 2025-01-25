# module 5

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands with optimized settings for right hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,              # Focus on single hand for better accuracy
    model_complexity=1,           # Balanced setting for accuracy and speed
    min_detection_confidence=0.6,  # Lowered for better initial detection
    min_tracking_confidence=0.6    # Lowered for smoother tracking
)
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize webcam with higher resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS for smoother capture

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

# Initialize drawing variables
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

current_color = colors['black']
thickness = 2
canvas = np.zeros((height, width, 3), dtype=np.uint8)
prev_x, prev_y = None, None
eraser_mode = False
drawing_mode = False

# Smoothing parameters
smooth_factor = 0.5
previous_points = []
max_points = 5

def smooth_coordinates(new_x, new_y):
    """Apply advanced smoothing to coordinates"""
    previous_points.append((new_x, new_y))
    if len(previous_points) > max_points:
        previous_points.pop(0)
    
    if len(previous_points) > 0:
        x = int(sum(p[0] for p in previous_points) / len(previous_points))
        y = int(sum(p[1] for p in previous_points) / len(previous_points))
        return x, y
    return new_x, new_y

def draw_color_palette(frame):
    """Draw color selection buttons"""
    for color_name, button in buttons.items():
        x, y, w, h = button['rect']
        color = button['color']
        cv2.rectangle(frame, (x, y), (x + w, h), color, -1)
        
        # Enhanced button visibility
        cv2.rectangle(frame, (x, y), (x + w, h), (255, 255, 255), 2)
        text_size = cv2.getTextSize(color_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        # Add text shadow for better visibility
        cv2.putText(frame, color_name, (text_x+1, text_y+1), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, color_name, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def check_drawing_gesture(hand_landmarks):
    """Enhanced gesture detection for right hand"""
    # Get relevant landmarks
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]
    index_mcp = hand_landmarks.landmark[5]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    wrist = hand_landmarks.landmark[0]

    # Check if hand is in proper position (right hand specific)
    is_right_position = wrist.x < index_tip.x  # For right hand

    # Check finger positions
    index_raised = index_tip.y < index_pip.y < index_mcp.y
    others_lowered = all([
        middle_tip.y > index_pip.y,
        ring_tip.y > index_pip.y,
        pinky_tip.y > index_pip.y
    ])

    return is_right_position and index_raised and others_lowered

while True:
    success, frame = cap.read()
    if not success:
        break
        
    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True
    
    # Overlay existing drawing
    mask = canvas.astype(bool).any(axis=2)
    frame[mask] = canvas[mask]
    
    # Draw color palette
    draw_color_palette(frame)
    
    # Process hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks with enhanced visibility
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Get index finger tip coordinates
            index_tip = hand_landmarks.landmark[8]
            x, y = int(index_tip.x * width), int(index_tip.y * height)
            
            # Apply smoothing
            x, y = smooth_coordinates(x, y)
            
            # Check for color selection
            if y < button_height:
                for color_name, button in buttons.items():
                    if x > button['rect'][0] and x < button['rect'][0] + button['rect'][2]:
                        if color_name == 'black':
                            eraser_mode = True
                        else:
                            eraser_mode = False
                        current_color = button['color']
                        previous_points.clear()  # Reset smoothing on color change
            
            # Check drawing gesture
            drawing_mode = check_drawing_gesture(hand_landmarks)
            
            if drawing_mode:
                if prev_x is None:
                    prev_x, prev_y = x, y
                else:
                    if eraser_mode:
                        # Enhanced eraser
                        cv2.circle(canvas, (x, y), 20, (0, 0, 0), -1)
                    else:
                        # Smoother line drawing
                        cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, thickness)
                    prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None
                previous_points.clear()

    # Add status indicator
    status_text = f"Mode: {'Eraser' if eraser_mode else 'Draw'} | Color: {[k for k,v in colors.items() if v == current_color][0]}"
    cv2.putText(frame, status_text, (10, height-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display frame
    cv2.imshow('Hand Drawing', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
    elif key == ord('+'):
        thickness = min(thickness + 1, 10)
    elif key == ord('-'):
        thickness = max(thickness - 1, 1)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
