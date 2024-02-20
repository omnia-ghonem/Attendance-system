import pickle
import cv2
import numpy as np
import face_recognition
import os

# Constants
PATH = 'Image_Attendance'
CSV_FILE = 'Attendance.csv'
CAMERA_INDEX = 0
FACE_RECOGNITION_THRESHOLD = 0.6
ENCODINGS_FILE = 'face_encodings.pickle'  # File to save face encodings
CLASS_NAMES_FILE = 'class_names.py'  # File to save class names

# Function to load images and extract class names
def load_images_and_class_names(path):
    images = []
    class_names = []

    # Get all folder names in the given path
    id_list = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]

    for i in id_list:
        folder_path = os.path.join(path, i)
        my_list = os.listdir(folder_path)

        for cl in my_list:
            cur_img = cv2.imread(os.path.join(folder_path, cl))
            images.append(cur_img)
            class_names.append(i)  # Use the folder name as the class name

    return images, class_names

# Function to find face encodings
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

# Function to save face encodings to a file
def save_encodings_to_file(encode_list):
    with open(ENCODINGS_FILE, 'wb') as file:
        pickle.dump(encode_list, file)

# Function to load face encodings from a file
def load_encodings_from_file():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as file:
            return pickle.load(file)
    else:
        return []

# Function to save class names to a file
def save_class_names_to_file(class_names):
    with open(CLASS_NAMES_FILE, 'w') as file:
        file.write("class_names = {}".format(repr(class_names)))

# Function to load class names from a file
def load_class_names_from_file():
    if os.path.exists(CLASS_NAMES_FILE):
        with open(CLASS_NAMES_FILE, 'r') as file:
            content = file.read()
            globals_ = {}
            locals_ = {}
            exec(content, globals_, locals_)
            return locals_['class_names'] if 'class_names' in locals_ else []
    else:
        return []
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
def detection(image):
    detected_faces = []

    if image is not None:
        try:
            image_gray = enhance_image(image)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

            margin = 10 # You can adjust the margin as needed (1 cm is approximately 28 pixels)

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
            cv2.imwrite('output_image.jpg', image)

        except Exception as e:
            print(f"Error: {e}")

    return detected_faces
# Modified main function to accept detected faces as input
def main():
    image_path = r"class.jpg"
    image = cv2.imread(image_path)
    detected_faces=detection(image)
    class_names = load_class_names_from_file()
    print(class_names)
    if not class_names:
        # If class names are not saved, extract and save them
        images, class_names = load_images_and_class_names(PATH)
        save_class_names_to_file(class_names)
        print('Class Names Saved')
    # Check if face encodings are saved
    encode_list_known = load_encodings_from_file()

    if not encode_list_known:
        # If face encodings are not saved, compute and save them
        encode_list_known = find_encodings(images)
        save_encodings_to_file(encode_list_known)
        print('Encoding Complete')
    
    try:
        # Set the desired number of columns for the display
        num_columns = 3

        # Calculate the number of rows needed based on the number of detected faces and columns
        num_rows = (len(detected_faces) + num_columns - 1) // num_columns

        # Calculate the dimensions of the canvas based on the first detected face
        canvas_height, canvas_width, _ = detected_faces[0].shape
        canvas_height *= num_rows
        canvas_width *= num_columns

        # Create a blank canvas with increased size for each image
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Initialize variables for placing images on the canvas
        row_index = 0
        col_index = 0

        # Resize all detected faces to the same dimensions
        resized_faces = [cv2.resize(face, (detected_faces[0].shape[1], detected_faces[0].shape[0])) for face in detected_faces]

        # Loop through the resized faces
        for img in resized_faces:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_cur_frame = face_recognition.face_locations(img_rgb)

            if face_cur_frame:
                encode_cur_frame = face_recognition.face_encodings(img_rgb, face_cur_frame)
                for encode_face, face_loc in zip(encode_cur_frame, face_cur_frame):
                    matches = face_recognition.compare_faces(encode_list_known, encode_face, tolerance=FACE_RECOGNITION_THRESHOLD)
                    face_dis = face_recognition.face_distance(encode_list_known, encode_face)
                    match_index = np.argmin(face_dis)

                    if matches[match_index]:
                        full_name = class_names[match_index].upper()
                        first_name, student_id = full_name.split('_')
                        print(first_name)
                        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                        cv2.putText(img, full_name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), 4)

                        # Calculate the position to place the image on the canvas
                        y_start = row_index * detected_faces[0].shape[0]
                        y_end = y_start + detected_faces[0].shape[0]
                        x_start = col_index * detected_faces[0].shape[1]
                        x_end = x_start + detected_faces[0].shape[1]

                        # Place the resized face image on the canvas
                        canvas[y_start:y_end, x_start:x_end, :] = img

                        # Update the column and row indices for the next image
                        col_index += 1
                        if col_index >= num_columns:
                            col_index = 0
                            row_index += 1

        # Show the image with recognized faces
        cv2.imshow('Recognized Faces', canvas)
        cv2.waitKey(0)

    except Exception as e:
        print(f"Error in upload_image: {e}")


if __name__ == "__main__":
    # Load the image and perform face detection

 main()


# Destroy all OpenCV windows
cv2.destroyAllWindows()
