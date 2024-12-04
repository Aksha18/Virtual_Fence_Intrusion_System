import cv2
import numpy as np
import face_recognition
import random
from collections import deque

def preprocess_frame(frame):
    # Apply Gaussian blur for noise reduction
    return cv2.GaussianBlur(frame, (5, 5), 0)

def augment_image(image, enable_augmentation=True):
    if not enable_augmentation:
        return image
    
    augment_type = random.choice(['flip', 'rotate', 'scale'])
    if augment_type == 'flip':
        image = cv2.flip(image, 1)
    elif augment_type == 'rotate':
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), random.randint(-30, 30), 1)
        image = cv2.warpAffine(image, M, (cols, rows))
    elif augment_type == 'scale':
        scale_factor = random.uniform(0.8, 1.2)  # Keep scaling closer to 1 for better face retention
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    return image

def extract_face_encodings(video_path, frame_skip=5):
    face_encodings_list = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video at {video_path}")
        return []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (640, 480))  # Reduce to lower resolution for efficiency
        augmented_frame = augment_image(frame, enable_augmentation=True)
        preprocessed_frame = preprocess_frame(augmented_frame)
        rgb_frame = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_encodings_list.extend(face_encodings)

    cap.release()
    return face_encodings_list

def recognize_face(frame, known_encodings, tolerance=0.5, distance_threshold=0.5, label_history=deque(maxlen=10)):
    frame = preprocess_frame(frame)
    frame = cv2.resize(frame, (650, 480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not face_encodings:  # No faces detected in the frame
        return frame  # Return the frame without any recognition

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        # Compare faces with the known encodings
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=tolerance)
        
        if len(matches) == 0:  # No matches found
            label = "Unknown"
        else:
            face_distances = face_recognition.face_distance(known_encodings, encoding)
            best_match_index = np.argmin(face_distances)

            # Only assign "Friend" if the distance is below the threshold
            if face_distances[best_match_index] < distance_threshold and matches[best_match_index]:
                label = "Friend"
            else:
                label = "Unknown"

        # Add label to the history
        label_history.append(label)
        
        # If the majority of recent labels are "Friend", then label as "Friend"
        if label_history.count("Friend") > len(label_history) // 2:
            label = "Friend"
        else:
            label = "Unknown"

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return frame


    
# Paths and video configurations
video_path = "E:/WIN_20241202_16_27_12_Pro.mp4"
your_face_encodings = extract_face_encodings(video_path, frame_skip=10)

if not your_face_encodings:
    print("No encodings found. Exiting.")
    exit()

cap = cv2.VideoCapture(0)  # Webcam capture
output_video_path = "output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (640, 480))

# Label history for smoothing
label_history = deque(maxlen=10)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = recognize_face(frame, your_face_encodings, tolerance=0.4, distance_threshold=0.6, label_history=label_history)
        out.write(annotated_frame)
        cv2.imshow("Face Recognition", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Program interrupted by user.")
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
