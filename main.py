import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Face Mesh module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Drawing specifications for landmarks
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Get the screen size for mouse control
screen_width, screen_height = pyautogui.size()

# Setup video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set video width
cap.set(4, 480)  # Set video height

# Number of frames to consider for smoothing
smoothing_frames = 5
cursor_positions = []

def get_smoothed_cursor_position(x, y):
    """Apply moving average smoothing to cursor position."""
    cursor_positions.append((x, y))
    if len(cursor_positions) > smoothing_frames:
        cursor_positions.pop(0)  # Remove the oldest position

    smoothed_x = sum(p[0] for p in cursor_positions) / len(cursor_positions)
    smoothed_y = sum(p[1] for p in cursor_positions) / len(cursor_positions)
    return smoothed_x, smoothed_y

# Indices of landmarks for upper and lower eyelids for one eye
upper_eyelid_indices = [386, 385, 384, 398]
lower_eyelid_indices = [374, 380, 381, 382]
eye_blink_threshold = 0.002

def get_eyelid_distance(landmarks, eye_indices):
    """Calculate the distance between upper and lower eyelids for one eye."""
    eye_points = np.array([(landmarks.landmark[point].x, landmarks.landmark[point].y) for point in eye_indices])
    upper_eyelid = eye_points[:len(eye_indices)//2]
    lower_eyelid = eye_points[len(eye_indices)//2:]
    return np.mean([np.linalg.norm(upper - lower) for upper, lower in zip(upper_eyelid, lower_eyelid)])


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to capture frame.")
        continue
    
    image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1]
            x, y = int(nose_tip.x * image.shape[1]), int(nose_tip.y * image.shape[0])
            
            screen_x, screen_y = get_smoothed_cursor_position(x, y)
            pyautogui.moveTo(screen_x, screen_y)

            eyelid_distance = get_eyelid_distance(face_landmarks, upper_eyelid_indices + lower_eyelid_indices)
            if eyelid_distance < eye_blink_threshold:
                pyautogui.click()

            mp_drawing.draw_landmarks(
                image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec)
    
    cv2.imshow('MediaPipe FaceMesh', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
