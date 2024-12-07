import cv2
import face_recognition
import numpy as np
import pickle
import os

# Function to augment the image
def augment_image(image, enable_augmentation=False):
    if enable_augmentation:
        return cv2.flip(image, 1)
    return image

# Process reference video to extract encodings with optimized frame sampling
def process_reference_video(video_path, frame_skip_factor=5):
    reference_encodings = []
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return reference_encodings

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{video_path}'.")
        return reference_encodings

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = max(1, fps // frame_skip_factor)
    print(f"Processing video with frame skip: {frame_skip}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        reference_encodings.extend(face_encodings)

    cap.release()
    print(f"Extracted {len(reference_encodings)} face encodings from the reference video.")
    return reference_encodings

# Save face encodings to a file
def save_encodings(encodings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(encodings, f)

# Load face encodings from a file
def load_encodings(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Encodings file '{file_path}' does not exist.")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Recognize faces in real-time webcam feed
def recognize_faces_in_webcam(known_encodings, tolerance=0.6, distance_threshold=0.5):
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
    if not cap.isOpened():
        print("Error: Unable to access webcam.")
        return

    print("Press 'q' to exit webcam recognition.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam. Exiting.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < distance_threshold:
                label = "Friend"
            else:
                label = "Unknown"

            top, right, bottom, left = face_location
            color = (0, 255, 0) if label == "Friend" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    reference_video_path = "E:/WIN_20241202_16_27_12_Pro.mp4"
    encodings_file = "face_encodings.pkl"

    try:
        print("Loading encodings...")
        your_face_encodings = load_encodings(encodings_file)
        print("Encodings successfully loaded from file.")
    except FileNotFoundError:
        print("Encodings file not found. Processing reference video...")
        your_face_encodings = process_reference_video(reference_video_path)
        if your_face_encodings:
            save_encodings(your_face_encodings, encodings_file)
            print("Encodings saved to file.")
        else:
            print("No face encodings were extracted. Exiting.")
            exit()

    print("Starting face recognition with webcam feed...")
    recognize_faces_in_webcam(your_face_encodings)
