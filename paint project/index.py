# module 1
import cv2
import mediapipe as mp
import numpy as np

try:
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # Start video capture
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from camera
        success, frame = cap.read()
        if not success:
            print("Failed to read camera frame")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = hands.process(rgb_frame)
        
        # Draw landmarks if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
        
        # Show the frame
        cv2.imshow('Hand Detection', frame)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Clean up
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
