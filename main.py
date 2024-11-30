import cv2
import face_recognition
import telegram
import numpy as np
import dlib
TOKEN = 'enter you token here'
bot = telegram.Bot(token=TOKEN)
chat_id =  your_chat_id

your_face = face_recognition.load_image_file('C:/Users/91956/Downloads/aksha pto.jpg')
your_face_encoding = face_recognition.face_encodings(your_face)[0]

cap = cv2.VideoCapture(0)

def send_alert():
    try:
        bot.send_message(chat_id=chat_id, text="Testing message from bot!")
        print("Alert sent to Telegram successfully!")
    except Exception as e:
        print(f"Error sending message: {e}")
try:
    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        if not ret:
            print("Failed to grab frame")
            break
        face_locations = face_recognition.face_locations(frame)  # Find face locations
        face_encodings = face_recognition.face_encodings(frame, face_locations)  # Encode detected faces

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces([your_face_encoding],face_encoding)

            if True in matches:
               print("Friend detected")
            else:
               print("Alert: Unkown person detected")
               send_alert()

        cv2.imshow('Video',frame)
        if cv2.waitKey(1) & 0xFF== ord('q'):
             break

except KeyboardInterrupt:  # Handle manual interruption like Ctrl+C
    print("Program interrupted by the user.")

finally:
  cap.release()  # Release the webcam resource
  cv2.destroyAllWindows()  # Close all OpenCV windows
