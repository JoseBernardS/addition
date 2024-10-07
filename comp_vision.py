import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def count_fingers(hand_landmarks):
    finger_tips = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP]
    finger_bases = [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP,
                    mp_hands.HandLandmark.PINKY_PIP]
    count = 0
    for tip, base in zip(finger_tips, finger_bases):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            count += 1
    # Special case for thumb
    if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_bases[0]].x:
        count += 1
    return count-1

cap = cv2.VideoCapture(0)
numbers = []
last_number = -1
last_number_time = 0
result_displayed = False
result = 0
last_estimation_time = 0
estimation_delay = 1  # 1 second delay

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)  # Flip the image horizontally for a selfie-view display
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        current_time = time.time()
        results = hands.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        finger_count = 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                finger_count = count_fingers(hand_landmarks)
        
        if current_time - last_estimation_time >= estimation_delay:
            last_estimation_time = current_time
            if finger_count > 0:
                numbers.append(finger_count)
                last_number = finger_count
                last_number_time = current_time
                result_displayed = False
        
        if not results.multi_hand_landmarks:
            if current_time - last_number_time > 2 and not result_displayed:
                result = sum(numbers)
                result_displayed = True
                numbers = []  # Reset numbers after calculating result

        # Display current finger count
        cv2.putText(image, f"Current: {finger_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 
                    2, lineType=cv2.LINE_AA)

        # Display numbers and result
        cv2.putText(image, f"Numbers: {numbers}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 
                    2, lineType=cv2.LINE_AA)
        if result_displayed:
            cv2.putText(image, f"Result: {result}", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 
                        2, lineType=cv2.LINE_AA)

        # Display a countdown for the next estimation
        time_to_next_estimation = max(0, estimation_delay - (current_time - last_estimation_time))
        cv2.putText(image, f"Next in: {time_to_next_estimation:.1f}s", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 
                    2, lineType=cv2.LINE_AA)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

cap.release()
cv2.destroyAllWindows()