import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Finger tip landmark indices for thumb, index, middle, ring, pinky
FINGER_TIPS = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    finger_count = 0

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark

        # Thumb: compare tip and ip landmarks x coordinate (different logic since thumb is sideways)
        if landmarks[FINGER_TIPS[0]].x > landmarks[FINGER_TIPS[0] - 1].x:
            finger_count += 1

        # Other fingers: tip y < pip y means finger is open (tip is higher than pip joint)
        for tip_id in FINGER_TIPS[1:]:
            if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                finger_count += 1

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f'Fingers: {finger_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Finger Counting - Press q to Quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
