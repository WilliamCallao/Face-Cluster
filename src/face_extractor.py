import cv2
import os

def extract_faces(input_path):
    faces_with_originals = []

    for filename in os.listdir(input_path):
        filepath = os.path.join(input_path, filename)
        image = cv2.imread(filepath)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in detected_faces:
            face = image[y:y+h, x:x+w]
            faces_with_originals.append((face, image))

    return faces_with_originals
