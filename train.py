import os
import numpy as np
import face_recognition
import cv2
import pickle

def enhance_image(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    enhanced_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return enhanced_img

def load_images_from_folder(folder):
    images, classNames, classIDs = [], [], []
    for person_folder in os.listdir(folder):
        person_path = os.path.join(folder, person_folder)
        if os.path.isdir(person_path):
            person_images = []
            for file in os.listdir(person_path):
                file_path = os.path.join(person_path, file)
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    img = cv2.imread(file_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        enhanced_img = enhance_image(img)
                        person_images.append(enhanced_img)
            if person_images:
                images.extend(person_images)  # Changed from append to extend to flatten the list
                classNames.extend([person_folder] * len(person_images))  # Replicate name for each image
                classIDs.extend([person_folder] * len(person_images))  # Replicate ID for each image

    return images, classNames, classIDs

def findEncodings(images, classNames):
    encodeList = []
    for index, img in enumerate(images):
        face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=1, model="cnn") # using CNN model
        face_encodings = face_recognition.face_encodings(img, face_locations)
        if face_encodings:
            average_encoding = np.mean(face_encodings, axis=0)
            encodeList.append(average_encoding)
            print(f"Encoded image {index+1}/{len(images)}: {classNames[index]}")
        else:
            print(f"No face found in image {index+1}/{len(images)}: {classNames[index]}")
    return encodeList


path = 'vision'
images, classNames, classIDs = load_images_from_folder(path)
encodeListKnown = findEncodings(images, classNames)
print('Encoding Complete')

with open('encodings.pkl', 'wb') as f:
    pickle.dump({"encodings": encodeListKnown, "names": classNames, "ids": classIDs}, f)
print("Encodings, names, and IDs saved to encodings.pkl")
