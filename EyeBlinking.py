import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
drawing_utils = mp.solutions.drawing_utils

# Eye landmark indices for blink detection (right and left eyes)
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    coords = [(int(landmarks.landmark[i].x * img_w), int(landmarks.landmark[i].y * img_h)) for i in eye_indices]
    # vertical distances
    A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    # horizontal distance
    C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    ear = (A + B) / (2.0 * C)
    return ear

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_h, img_w = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, img_w, img_h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, img_w, img_h)
        avg_ear = (left_ear + right_ear) / 2.0

        # Threshold for blink detection (tune if needed)
        BLINK_THRESHOLD = 0.25

        if avg_ear < BLINK_THRESHOLD:
            cv2.putText(frame, "Blinking", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Eyes Open", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Blink Detection - Press q to Quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
