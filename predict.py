import cv2
import numpy as np
import face_recognition
import pickle
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

def markAttendance(name, id, dtString):
    with open("Attendance.csv", "r+") as f:
        existing_entries = f.readlines()
        name_id_dt = f"{name},{id},{dtString}\n"
        if name_id_dt not in existing_entries:
            f.write(name_id_dt)

def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    root.destroy()
    return file_path

# Load known face encodings, names, and IDs
def enhance_image(img):
    # Apply histogram equalization to enhance image contrast
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    enhanced_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return enhanced_img

with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)
encodeListKnown = data["encodings"]
classNames = data["names"]
classIDs = data["ids"]



def enhance_image(image):
    
    # Convert the image to grayscale for histogram equalization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    enhanced_image = cv2.equalizeHist(gray)

    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(enhanced_image, (0, 0), 3)
    sharp = cv2.addWeighted(enhanced_image, 1.5, blurred, -0.5, 1)

    return sharp

def detection(image,margin):
    detected_faces = []

    if image is not None:
        try:
            image_gray = enhance_image(image)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.02, minNeighbors=3, minSize=(10, 10))

            margin = margin # You can adjust the margin as needed (1 cm is approximately 28 pixels)

            for i, (x, y, w, h) in enumerate(faces):
                # Add margin to the face region
                x -= margin
                y -= margin
                w += 2 * margin
                h += 2 * margin

                # Ensure the coordinates are within the image boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(image.shape[1] - x, w)
                h = min(image.shape[0] - y, h)

                # Extract the face region with margin
                face_region = image[y:y+h, x:x+w]
                detected_faces.append(face_region)

                # Draw a rectangle around the detected face with margin

            # Save the modified image with rectangles around the detected faces

        except Exception as e:
            print(f"Error: {e}")

    return detected_faces

image_file_path = select_image()
if image_file_path:
    img = cv2.imread(image_file_path)
    detected_faces = detection(img,6)

    for detected_face in detected_faces:
        img = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
        if img is not None:
            # Enhance the input image
            # img = enhance_image(img_rgb)

            # imgS = cv2.resize(img, (0, 0), None, 1, 1)
            print('before location')
            # Use both HOG and CNN models for face detection
            faceCurFrameHOG = face_recognition.face_locations(img, number_of_times_to_upsample=1, model="hog")
            print('after location')

            # Combine the results of both models
            faceCurFrame = faceCurFrameHOG 

            encodeCurFrame = face_recognition.face_encodings(img, faceCurFrame)

            for encodeFace, face_loc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex]
                    id = classIDs[matchIndex]
                    print("Recognized:", name, "ID:", id)
                    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                    font_scale=4
                    cv2.rectangle(detected_face, (x1, y1), (x2, y2), (0, 0, 200), 4)
                    thickness = 1

                    # cv2.putText(detected_face, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 200), thickness)

                    now = datetime.now()
                    dtString = now.strftime("%H:%M:%S:%Y-%m-%d")
                    markAttendance(name, id, dtString)
                    resized_image = cv2.resize(detected_face, (600, 600))
                    cv2.imshow(f"Recognized Face - {name}", resized_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()


    else:
        print("Error: Unable to read the loaded image.")

else:
    print("No image file selected.")
