import tkinter as tk
from tkinter import SINGLE, Button, Entry, Image, Listbox, Label, END, PhotoImage, StringVar, Tk, Frame, Toplevel
from tkinter import messagebox
from tkinter import filedialog
import cv2
import csv
import numpy as np
import face_recognition
import os
from datetime import datetime
import mysql.connector
from PIL import Image, ImageTk
import pickle
import pandas as pd
from datetime import datetime
import openpyxl

from frame2 import frame_video  # Correct import statement


# Constants defining file paths, camera index, and face recognition threshold
PATH = 'Image_Attendance'
CSV_FILE = 'Attendance.csv'
ENCODINGS_FILE = 'encodings.pkl'  # File to save face encodings
CAMERA_INDEX = 0
FACE_RECOGNITION_THRESHOLD = 0.6
CLASS_NAMES_FILE = 'class_names.py'  # File to save class names

# attended_students_count=0
# absent_students_count=0
# table_name = 'Students'
# database_name = 'attendance'
port=3306
dbhost = 'btbrazwnstmtza82arcm-mysql.services.clever-cloud.com'
dbuser ='uu6tbaokps9llpam'
dbpass = '3bDQ3bD4upyn4t4o4iFi'
dbname = 'btbrazwnstmtza82arcm'

# try:

# Set to keep track of registered IDs for the current day
registered_ids_today = set()
def upload_and_process():
    # Ask user to select a video file
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if not file_path:
        return  # User canceled or didn't select a file
    direction = f'D:/Ejust 4th year/first term/CSE 429 Computer Vision and Pattern Recognition/final project/frame_videos/{id}'

    # Create the directory if it doesn't exist
    os.makedirs(direction, exist_ok=True)

    # Process the video frames
    frame_folder = os.path.normpath(direction)
    frame_video(file_path, frame_folder)
    status_label.config(text=f"Video frames is uploaded")

def get_database_connection():
    return mysql.connector.connect(
        host=dbhost,
        database=dbname,
        user=dbuser,
        password=dbpass,
        port=port
    )
def register():
    global register_screen
    register_screen = Toplevel(window)
    register_screen.title("Register")
    register_screen.geometry("1000x500")
 
    # global username
    # global password
    # global username_entry
    # global password_entry
    username = StringVar()
    password = StringVar()
    Email = StringVar()
 
    Label(register_screen, text="Please enter details below", bg="grey").pack()
    Label(register_screen, text="").pack()
    username_lable = Label(register_screen, text="Username * ")
    username_lable.pack()

    username_entry = Entry(register_screen, textvariable=username)
    username_entry.pack()

    password_lable = Label(register_screen, text="Password * ")
    password_lable.pack()

    password_entry = Entry(register_screen, textvariable=password, show='*')
    password_entry.pack()

    Email_lable = Label(register_screen, text="Email * ")
    Email_lable.pack()

    Email_entry = Entry(register_screen, textvariable=Email)
    Email_entry.pack()

    Label(register_screen, text="").pack()
    Button(register_screen, text="Register", width=10, height=1, bg="lightblue", command =lambda:  register_user(username ,password,Email,username_entry,password_entry,Email_entry)).pack()
 
 
# Designing window for login 
 
def login():
    global login_screen
    login_screen = Toplevel(window)
    login_screen.title("Login")
    login_screen.geometry("1000x500")
    Label(login_screen, text="Please enter details below to login").pack()
    Label(login_screen, text="").pack()
 


    Email_verify = StringVar()
    password_verify = StringVar()
 

 
    Label(login_screen, text="Email * ").pack()
    Email_login_entry = Entry(login_screen, textvariable=Email_verify)
    Email_login_entry.pack()

    Label(login_screen, text="").pack()

    Label(login_screen, text="Password * ").pack()
    password_login_entry = Entry(login_screen, textvariable=password_verify, show= '*')
    password_login_entry.pack()

    Label(login_screen, text="").pack()

    # send the Email_login_entry , password_login_entry to delete the content of the entry after click login
    Button(login_screen, text="Login", width=10, height=1, command =lambda: login_verify(Email_verify,password_verify,Email_login_entry,password_login_entry)).pack() 
 
# Implementing event on register button
 
def register_user(username ,password,Email,username_entry,password_entry,Email_entry):
    dbconnection=get_database_connection()
    db = dbconnection.cursor()

    username_info = username.get()
    username_info=username_info.strip()

    password_info = password.get()
    password_info=password_info.strip()

    Email_info=Email.get()
    Email_info=Email_info.strip()

    if(len(str(username_info)) <0):
        
        messagebox.showerror("ERROR", "Must enter user name")
        return

    
    if(len(password_info) <0 ):
        messagebox.showerror("ERROR", "Must enter password")
        return
    
    if(len(Email_info) <0):
        messagebox.showerror("ERROR", "Must enter Email")
        return
    # file = open(username_info, "w")
    # file.write(username_info + "\n")
    # file.write(password_info)
    # file.close()
    exist_user = f"SELECT Email FROM USERS WHERE Email = '{Email_info}';"

    db.execute(exist_user)
    result = db.fetchall()
    # print(result)
    if (result):
        messagebox.showerror("ERROR", "this user exist ")
        username_entry.delete(0, END)
        password_entry.delete(0, END)
        Email_entry.delete(0, END)
        return
    else:
        add_user = f"INSERT INTO USERS (UserName ,Email  ,Password) VALUES(%s,%s,%s);"
        Values = (username_info,Email_info,password_info)

        db.execute(add_user,Values)
        dbconnection.commit()

        result = db.fetchall()
        Label(register_screen, text="Registration Success", fg="green", font=("calibri", 11)).pack()
        register_screen.destroy()

        username_entry.delete(0, END)
        password_entry.delete(0, END)
        Email_entry.delete(0, END)
 
 
# Implementing event on login button 
 
def login_verify(Email_verify,password_verify,Email_login_entry,password_login_entry):
    global Email1
    dbconnection=get_database_connection()
    db = dbconnection.cursor()
    Email1 = Email_verify.get()
    Email1=Email1.strip()

    password1 = password_verify.get()
    password1=password1.strip()
    if(len(str(Email1)) <0):
        
        messagebox.showerror("ERROR", "Must enter  Email")
        return

    
    if(len(password1) <0 ):
        messagebox.showerror("ERROR", "Must enter password")
        return



    email_exists= f"SELECT Email FROM USERS WHERE Email = '{Email1}';"
    db.execute(email_exists)
    result = db.fetchall()
    if(result):

        add_user = f"SELECT Password FROM USERS WHERE Email = '{Email1}';"
        db.execute(add_user)
        result = db.fetchall()
        password_registered = result[0][0]
        if(password1==password_registered):
            Label(login_screen, text="Login Success", fg="green", font=("calibri", 11)).pack()
            Email_login_entry.delete(0, END)
            password_login_entry.delete(0, END)
            login_screen.destroy()

            switch_to_home_page(frame_second, cap, db_connection)

        else:
            messagebox.showerror("ERROR", "password is incorrect")
            Email_login_entry.delete(0, END)
            password_login_entry.delete(0, END)
            return
    else:
            messagebox.showerror("ERROR", "This email doesn't exist")
            Email_login_entry.delete(0, END)
            password_login_entry.delete(0, END)
            return
    # print("password : ",id)
 
# Designing popup for login success
 
def login_sucess():
    global login_success_screen
    login_success_screen = Toplevel(login_screen)
    login_success_screen.title("Success")
    login_success_screen.geometry("330x100")
    Label(login_success_screen, text="Login Success").pack()
    Button(login_success_screen, text="OK", command=delete_login_success).pack()
 
# Designing popup for login invalid password
 
def password_not_recognised():
    global password_not_recog_screen
    password_not_recog_screen = Toplevel(login_screen)
    password_not_recog_screen.title("Success")
    password_not_recog_screen.geometry("330x100")
    Label(password_not_recog_screen, text="Invalid Password ").pack()
    Button(password_not_recog_screen, text="OK", command=delete_password_not_recognised).pack()
 
# Designing popup for user not found
 
def user_not_found():
    global user_not_found_screen
    user_not_found_screen = Toplevel(login_screen)
    user_not_found_screen.title("Success")
    user_not_found_screen.geometry("330x100")
    Label(user_not_found_screen, text="User Not Found").pack()
    Button(user_not_found_screen, text="OK", command=delete_user_not_found_screen).pack()
 
# Deleting popups
 
def delete_login_success():
    login_success_screen.destroy()
 
 
def delete_password_not_recognised():
    password_not_recog_screen.destroy()
 
 
def delete_user_not_found_screen():
    user_not_found_screen.destroy()
 
 
# Designing Main(first) window
 

def download_attendance_to_excel(Name_of_table, connection):
    try:
        with connection.cursor() as cursor:
            select_attendance_data = f"SELECT * FROM {Name_of_table} WHERE DATE(DayOfAttend) = CURDATE();"
            cursor.execute(select_attendance_data)
            result = cursor.fetchall()

            # Create a DataFrame from the result
            df = pd.DataFrame(result, columns=['Name', 'person_id', 'DayOfAttend'])

            # Convert 'DayOfAttend' column to string format
            df['DayOfAttend'] = df['DayOfAttend'].astype(str)

            current_date = datetime.now().strftime("%Y-%m-%d")

            # Save the DataFrame to an Excel file with the current date in the name
            excel_filename = f'{Name_of_table}_attendance_{current_date}.xlsx'
            df.to_excel(excel_filename, index=False)

            messagebox.showinfo("Success", f"Attendance data downloaded to {excel_filename}")
    except Exception as e:
        messagebox.showerror("Error", f"Error downloading attendance: {str(e)}")

# ...
def user():
    global id,Email1
    connection = get_database_connection()
    cursor = connection.cursor()

    user_email = Email1
    select_id = f"SELECT ID FROM USERS WHERE Email = '{user_email}';"
    cursor.execute(select_id)
    result = cursor.fetchall()
    id = result[0][0]
    print(id)
    connection.close()

# Function to mark attendance in CSV file and MySQL table
def mark_attendance(name, person_id, Name_of_table, connection):
    global registered_ids_today

    # Uncomment the following section if you want to update the MySQL table
    if not is_id_registered_in_mysql(person_id, Name_of_table, connection):
        try:
            with connection.cursor() as cursor:
                insert_in_table = f"INSERT INTO {Name_of_table} (Name, person_id) VALUES(%s, %s);"
                values = (name, person_id)
                cursor.execute(insert_in_table, values)
            connection.commit()

        except Exception as e:
            print(f"Error marking attendance: {e}")
def absent_students( Name_of_table, connection):
    global absent_students_count
    Name_of_table=f'{Name_of_table}_all_people'
    try:
        with connection.cursor() as cursor:
            display_students_in_table = f"SELECT COUNT(*) FROM {Name_of_table};"
            cursor.execute(display_students_in_table)
            result = cursor.fetchall()
            print(result[0][0])
            absent_students_count=result[0][0]-theList.size()

    except Exception as e:
        print(f"Error displaying students: {e}")
# Uncomment the following function if you want to check if the ID is already registered in the MySQL table
def is_id_registered_in_mysql(person_id, Name_of_table, connection):
    try:
        with connection.cursor() as cursor:
            select_student_if_exist = f"SELECT * FROM {Name_of_table} WHERE person_id = %s AND DATE(DayOfAttend) = CURDATE();"
            ID = (person_id,)
            cursor.execute(select_student_if_exist, ID)
            result = cursor.fetchall()
        return bool(result)
    except Exception as e:
        print(f"Error checking registration: {e}")
        return False

# Function to load images and extract class names from the specified path
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

# Function to save face encodings to a file
def save_encodings_to_file(encodeListKnown,classNames,classIDs):
    with open('encodings.pkl', 'wb') as f:
        pickle.dump({"encodings": encodeListKnown, "names": classNames, "ids": classIDs}, f)
    print("Encodings, names, and IDs saved to encodings.pkl")

# Function to load face encodings from a file
def load_encodings_from_file():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        encodeListKnown = data["encodings"]
        classNames = data["names"]
        classIDs = data["ids"]
        return encodeListKnown,classNames,classIDs
    else:
        return []
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
# Function to create the attendance database if it does not exist
# def create_to_database(database_name):
#     connect = mysql.connector.connect(host='localhost', password='', user='root')
#     my_cursor = connect.cursor()
#     my_cursor.execute("SHOW DATABASES;")
#     databases = my_cursor.fetchall()  
#     if ('attendance',) in databases:
#         print('attendance database exists')
#     else:
#         create_database = f"CREATE DATABASE {database_name};"
#         my_cursor.execute(create_database)
#     connect.commit()
#     connect.close()

# Function to create the attendance table if it does not exist
# def create_table(table):
#     connect = mysql.connector.connect(host='localhost', password='', user='root', database='attendance')
#     db = connect.cursor()
#     table_be_created = f"CREATE TABLE IF NOT EXISTS {table} (FirstName varchar(255) NOT NULL, Stu_id int(11) NOT NULL, DayOfAttend timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp());"
#     db.execute(table_be_created)   
#     connect.commit()
#     connect.close()

# Function to display attended students today in the GUI Listbox
def show_register_data(Name_of_table, connection, theList):
    try:
        with connection.cursor() as cursor:
            display_attended_student_today = f"SELECT * FROM {Name_of_table} WHERE DATE(DayOfAttend) = CURDATE();"
            cursor.execute(display_attended_student_today)
            result = cursor.fetchall()

            # Assuming you have a Listbox named 'theList' for displaying data
            theList.delete(0, END)  # Clear the existing data in the Listbox
            for record in result:
                formatted_data = f"{record[0]} - {record[1]} - {record[2]}"
                theList.insert(END, formatted_data)
    except Exception as e:
        print(f"Error displaying attended students: {e}")

# Function to remove a selected student record from the MySQL table
def remove_student(Name_of_table, connection, theList):
    global absent_students_count
    selected_index = theList.curselection()  # Get the index of the selected item in the Listbox
    if not selected_index:
        messagebox.showinfo("INFO", "Please select a record to remove.")
        return
        # Get the content of the selected item (assuming it contains the necessary information)
    selected_item = theList.get(selected_index)
    
    # Extract necessary information from the selected item (customize based on your Listbox content)
    # Assuming the format is "studentId - courseId - grade - mark"
    selected_data = selected_item.split(" - ")
    ID = selected_data[1]
    print(ID)
    try:
        with connection.cursor() as cursor:
            # Execute the DELETE query to remove the selected record
            delete_from_table = f"DELETE FROM {Name_of_table} WHERE person_id=%s"
            values = (ID,)
            cursor.execute(delete_from_table, values)
        connection.commit()  # Save (commit) the changes
    except Exception as e:
        print(f"Error removing student: {e}")
    finally:
            show_register_data(Name_of_table, connection, theList)
            absent_students(Name_of_table, connection)

            attended_label.config(text=f"Attended: {theList.size()}")
            absent_label.config(text=f"Absent: {absent_students_count}")
            upload_button = tk.Button(frame_second, text='Upload Image', command=lambda: upload_image(Name_of_table, encodeListKnown, classNames,classIDs, connection, label, theList))
            upload_button.grid(row=4, column=0, padx=10, pady=5)
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
                x += margin
                y += margin
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
def no_file_path(selected_table, encodeListKnown, classNames,classIDs, connection, label, theList):
    global file_path
    file_path= r'D:\Ejust 4th year\first term\CSE 429 Computer Vision and Pattern Recognition\Face_Recognition\transparent.png'
    img = cv2.imread(file_path)

    detected_faces = detection(img,6)
    
    for detected_face in detected_faces:
        imgs = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
        if img is not None:

            faceCurFrameHOG = face_recognition.face_locations(imgs, number_of_times_to_upsample=1, model="hog")

            # Combine the results of both models
            faceCurFrame = faceCurFrameHOG 

            encodeCurFrame = face_recognition.face_encodings(imgs, faceCurFrame)

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
                    mark_attendance(name, id, selected_table, connection)

    show_register_data(selected_table, connection, theList)
    absent_students(selected_table, connection)
    
    frame_second = cv2.resize(img, (600, 600))
    img = cv2.cvtColor(frame_second, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    attended_label.config(text=f"Attended: {theList.size()}")
    absent_label.config(text=f"Absent: {absent_students_count}")



def upload_image(selected_table, encodeListKnown, classNames,classIDs, connection, label, theList):
    try:
        file_path = filedialog.askopenfilename(title="Select an Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            img = cv2.imread(file_path)
            detected_faces = detection(img,6)

            for detected_face in detected_faces:
                imgs = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
                if imgs is not None:
                    # Enhance the input image
                    # img = enhance_image(img_rgb)

                    # imgS = cv2.resize(img, (0, 0), None, 1, 1)
                    print('before location')
                    # Use both HOG and CNN models for face detection
                    faceCurFrameHOG = face_recognition.face_locations(imgs, number_of_times_to_upsample=1, model="hog")
                    print('after location')

                    # Combine the results of both models
                    faceCurFrame = faceCurFrameHOG 

                    encodeCurFrame = face_recognition.face_encodings(imgs, faceCurFrame)

                    for encodeFace, face_loc in zip(encodeCurFrame, faceCurFrame):
                        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                        matchIndex = np.argmin(faceDis)

                        if matches[matchIndex]:
                            name = classNames[matchIndex]
                            id = classIDs[matchIndex]
                            print("Recognized:", name, "ID:", id)
                            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                            thickness = 10
                            cv2.rectangle(detected_face, (x1, y1), (x2, y2), (0, 0, 200), thickness)
                            mark_attendance(name, id, selected_table, connection)

            show_register_data(selected_table, connection, theList)
            absent_students(selected_table, connection)
            
            frame_second = cv2.resize(img, (600, 600))
            img = cv2.cvtColor(frame_second, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)
            attended_label.config(text=f"Attended: {theList.size()}")
            absent_label.config(text=f"Absent: {absent_students_count}")

    except Exception as e:
        print(f"Error in upload_image: {e}")


    # Reset the flag after handling the button click
# Function to update the video frame for face recognition
# def update_frame(Name_of_table, cap, encode_list_known, class_names, connection, label, theList, window):
#     global registered_ids_today,full_name,attended_students_count, absent_students_count
#     full_name=None



#     try:
#         success, img = cap.read()
#         img_s = cv2.resize(img, (0, 0), None, 0.25, 0.25)
#         img_rgb = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

#         face_cur_frame = face_recognition.face_locations(img)
#         if face_cur_frame:
#             # Face recognition code
#             encode_cur_frame = face_recognition.face_encodings(img, face_cur_frame)
#             for encode_face, face_loc in zip(encode_cur_frame, face_cur_frame):
#                 matches = face_recognition.compare_faces(encode_list_known, encode_face, tolerance=FACE_RECOGNITION_THRESHOLD)
#                 face_dis = face_recognition.face_distance(encode_list_known, encode_face)
#                 match_index = np.argmin(face_dis)

#                 if matches[match_index]:
#                     print('reach')
#                     full_name = class_names[match_index].upper()
#                     print(full_name)
#                     first_name, student_id = full_name.split('_')
#                     y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
#                     cv2.putText(img, full_name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
#                     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), 4)
#                     mark_attendance(first_name, student_id, Name_of_table, connection)

#         remove_student(Name_of_table, connection, theList)  # Remove student regardless of face recognition
#         show_register_data(Name_of_table, connection, theList)
#         absent_students(Name_of_table, connection)
#         frame_second = cv2.resize(img, (600, 600))
#         img = cv2.cvtColor(frame_second, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(img)
#         imgtk = ImageTk.PhotoImage(image=img)
#         label.imgtk = imgtk
#         label.configure(image=imgtk)
#         attended_label.config(text=f"Attended: {theList.size()}")
#         absent_label.config(text=f"Absent: {absent_students_count}")
#     except Exception as e:
#         print(f"Error in update_frame: {e}")
#     finally:
#         window.after(10, update_frame, Name_of_table, cap, encode_list_known, class_names, connection, label, theList, window)

# create table that user insert its name 
def create_table_indatabase(create_table_name, frame_second, cap, connection):
    if len(create_table_name.get()) < 1:
        messagebox.showerror("ERROR", "You must enter the name of the table")
        return

    try:
        with connection.cursor() as cursor:
            # Check if the table already exists
            table_name = f'{create_table_name.get()}_{id}'
            All_People_table_name = f'{create_table_name.get()}_{id}_all_people'

            check_table_exist = f"SHOW TABLES LIKE '{table_name}';"
            cursor.execute(check_table_exist)
            result = cursor.fetchall()
            print(result)
            if result:
                # Table already exists, display a message
                messagebox.showinfo("Table Exists", f"Table '{create_table_name.get()}' already exists")
            else:
                # Table does not exist, create it
                created_table = cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (Name varchar(255) NOT NULL , person_id varchar(255) NOT NULL PRIMARY KEY , DayOfAttend timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp());")
                cursor.execute(created_table)
                connection.commit()
                created_table_all = cursor.execute(f"CREATE TABLE IF NOT EXISTS {All_People_table_name} (Name varchar(255) NOT NULL , person_id varchar(255) NOT NULL PRIMARY KEY);")
                cursor.execute(created_table_all)
                connection.commit()
                messagebox.showinfo("Success", f"Table '{create_table_name.get()}' created successfully")

    except Exception as e:
        messagebox.showerror("ERROR", f"An error occurred: {str(e)}")
        return

    finally:
        switch_to_home_page(frame_second, cap, connection)

def Add_person(create_Person_Name,create_Person_id, frame_second, cap, connection):
    selected_table = tableList.curselection()  # Get the index of the selected item in the Listbox
    if not selected_table:
        messagebox.showinfo("INFO", "Please select a table")
        return
    # Get the content of the selected item (assuming it contains the necessary information)
    selected_table = tableList.get(selected_table) 
 
    if len(create_Person_Name.get()) < 1:
        messagebox.showerror("ERROR", "You must enter the name person")
        return
    if len(create_Person_id.get()) < 1:
        messagebox.showerror("ERROR", "You must enter the id of person")
        return
    All_People_table_name = f'{selected_table}_all_people'
    create_Person_Name=create_Person_Name.get()
    create_Person_id=create_Person_id.get()
    
    try:
        with connection.cursor() as cursor:
            # Execute the DELETE query to remove the selected record
            add_person = f"INSERT INTO {All_People_table_name} (Name, person_id) VALUES(%s, %s);"
            values = (create_Person_Name, create_Person_id)
            cursor.execute(add_person, values)
        connection.commit()
    except Exception as e:
        messagebox.showerror("ERROR", f"Error: {str(e)}")
    finally:
        switch_to_home_page(frame_second, cap, connection)
def Students_in_table(frame_second, connection):
    selected_table = tableList.curselection()  # Get the index of the selected item in the Listbox
    if not selected_table:
        messagebox.showinfo("INFO", "Please select a record to show students.")
        return
    # Get the content of the selected item (assuming it contains the necessary information)
    selected_table = tableList.get(selected_table)
    selected_table=f'{selected_table}_all_people'
    # Create a new top-level window
    students_window = Toplevel(frame_second)
    students_window.title(f"Students in {selected_table}")
    
    try:
        with connection.cursor() as cursor:
            display_students_in_table = f"SELECT * FROM {selected_table};"
            cursor.execute(display_students_in_table)
            result = cursor.fetchall()

            # Assuming you have a Listbox named 'studentsList' for displaying data
            studentsList = Listbox(students_window, selectmode=tk.SINGLE)
            studentsList.grid(row=0, column=0, padx=10, pady=10)
            studentsList.config(width=100, height=30)

            for record in result:
                formatted_data = f"{record[0]}-{record[1]}"
                studentsList.insert(END, formatted_data)
    except Exception as e:
        print(f"Error displaying students: {e}")

# Function to delete a table from the database
def Delete_table_indatabase(frame_second, cap, connection):
    selected_table = tableList.curselection()  # Get the index of the selected item in the Listbox
    if not selected_table:
        messagebox.showinfo("INFO", "Please select a record to remove.")
        return
    # Get the content of the selected item (assuming it contains the necessary information)
    selected_table = tableList.get(selected_table)

    try:
        with connection.cursor() as cursor:
            # Execute the DELETE query to remove the selected record
            delete_table = f"DROP TABLE {selected_table};"
            cursor.execute(delete_table)
            connection.commit()  # Save (commit) the changes
        tables_in_database(connection)  # Not necessary but makes it fast to appear the change
    except Exception as e:
        messagebox.showerror("ERROR", f"Error: {str(e)}")
    finally:
        switch_to_home_page(frame_second, cap, connection)

# Function to retrieve tables in the database and show in the Listbox
def tables_in_database(connection):
    try:
        with connection.cursor() as cursor:
            select_database = f"USE {dbname};"
            cursor.execute(select_database)

            display_tables_in_database = f"SHOW TABLES LIKE '%{id}';"
            cursor.execute(display_tables_in_database)
            result = cursor.fetchall()

            # Assuming you have a Listbox named 'tableList' for displaying data
            tableList.delete(0, END)  # Clear the existing data in the Listbox
            for record in result:
                formatted_data = f"{record[0]}"
                tableList.insert(END, formatted_data)
    except Exception as e:
        print(f"Error displaying tables: {e}")



def select_specific_table_to_show(cap, connection):
    selected_table = tableList.curselection()  # Get the index of the selected item in the Listbox
    if not selected_table:
        message = f"Please select a table to take {dbname}."
        messagebox.showinfo(f"INFO", message)
        return
    # Get the content of the selected item (assuming it contains the necessary information)
    selected_table = tableList.get(selected_table)
    switch_to_second_page(selected_table, cap, connection)

def hide_frame(frame):
    frame.grid_forget()

def release_capture():
    global cap
    if cap:
        cap.release()

# Function to switch to the home page
def switch_to_home_page(frame_second, cap, connection):
    global status_label
    user()
    release_capture()  # Release video capture when switching to the home page
    create_table_name_var = StringVar()  # Variable to store the Entry widget value
    create_Person_Name = StringVar()  # Variable to store the Entry widget value
    create_Person_id=StringVar() 
    hide_frame(frame_second)
    frame_home.tkraise()
    tables_in_database(connection)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    create_table_name_var.set("")  # Clear the Entry widget

    tableName = Label(frame_home, text="Table_name:")
    tableName.place(relx=0.6, rely=0, relheight=0.1, relwidth=0.1)

    input_of_table_name = Entry(frame_home, textvariable=create_table_name_var)
    input_of_table_name.place(relx=0.7, rely=0, relheight=0.1, relwidth=0.2)

    create_table = tk.Button(frame_home, text='Create Table', command=lambda: create_table_indatabase(create_table_name_var, frame_second, cap, connection))
    create_table.place(relx=0.6, rely=0.2, relheight=0.08, relwidth=0.2)

    delete_table = tk.Button(frame_home, text='Delete Table', command=lambda: Delete_table_indatabase(frame_second, cap, connection))
    delete_table.place(relx=0.7, rely=0.3, relheight=0.08, relwidth=0.08)

    switch_second = tk.Button(frame_home, text='Attendance Page using upload', command=lambda: select_specific_table_to_show(cap, connection))
    switch_second.place(relx=0.7, rely=0.4, relheight=0.08, relwidth=0.15)


    Person_Name = Label(frame_home, text="Person Name:")
    Person_Name.place(relx=0.6, rely=0.6, relheight=0.1, relwidth=0.1)

    input_of_Person_Name = Entry(frame_home, textvariable=create_Person_Name)
    input_of_Person_Name.place(relx=0.7, rely=0.6, relheight=0.1, relwidth=0.2)

    Person_id = Label(frame_home, text="Person id:")
    Person_id.place(relx=0.6, rely=0.7, relheight=0.1, relwidth=0.1)

    input_of_Person_id = Entry(frame_home, textvariable=create_Person_id)
    input_of_Person_id.place(relx=0.7, rely=0.7, relheight=0.1, relwidth=0.2)

    add_person_button = tk.Button(frame_home, text='Add Person', command=lambda: Add_person(create_Person_Name, create_Person_id, frame_second, cap, connection))
    add_person_button.place(relx=0.7, rely=0.8, relheight=0.08, relwidth=0.08)

    stud_in_table = tk.Button(frame_home, text='Show students in table', command=lambda: Students_in_table(frame_second, connection))
    stud_in_table.place(relx=0.7, rely=0.9, relheight=0.08, relwidth=0.08)
    
    upload_button = Button(frame_home, text="Upload Video", command=upload_and_process)
    upload_button.place(relx=0.3, rely=0.7, relheight=0.08, relwidth=0.08)

    status_label = Label(frame_home, text="")
    status_label.place(relx=0.3, rely=0.8, relheight=0.08, relwidth=0.1)
    window.after(10, lambda: tables_in_database(connection))

def switch_to_login(frame_home,frame_second):

    hide_frame(frame_second)
    hide_frame(frame_home)

    frame_login.tkraise()
    label = tk.Label(frame_login,text="Select Your Choice", bg="lightblue", width="300", height="2", font=("Calibri", 13))
    label.place(relx=0.4, rely=0.2, relheight=0.1, relwidth=0.1)
    login_button = tk.Button(frame_login, text='Login',  command = login)
    login_button.place(relx=0.4, rely=0.4, relheight=0.04, relwidth=0.08)
    register_button = tk.Button(frame_login, text='Register',  command = register)
    register_button.place(relx=0.4, rely=0.5, relheight=0.04, relwidth=0.08)
 

# Function to switch to the second page
def switch_to_second_page(selected_table, cap, connection):
    release_capture()  # Release video capture when switching to home page

    hide_frame(frame_home)

    frame_second.tkraise()
    switch_home = tk.Button(frame_second, text='Tables Page', command=lambda: switch_to_home_page(frame_second, cap, db_connection))
    switch_home.grid(row=2, column=0, padx=10, pady=5) 
    upload_button = tk.Button(frame_second, text='Upload Image', command=lambda: upload_image(selected_table, encodeListKnown, classNames,classIDs, connection, label, theList))
    upload_button.grid(row=4, column=0, padx=10, pady=5)
                
    remove = tk.Button(frame_second, text='remove student', command=lambda: remove_student(selected_table, connection, theList))
    remove.grid(row=4, column=3, padx=10, pady=5)
    no_file_path(selected_table, encodeListKnown, classNames,classIDs, connection, label, theList)    # update_frame(selected_table, cap, encode_list_known, class_names, connection, label, theList, window)
    download_button = tk.Button(frame_second, text='Download Attendance', command=lambda: download_attendance_to_excel(selected_table, db_connection))
    download_button.grid(row=3, column=0, padx=10, pady=5)  #

if __name__ == "__main__":
    # Load images and class names
    cap = cv2.VideoCapture(CAMERA_INDEX)

    # Check if face encodings are saved
    encodeListKnown,classNames,classIDs = load_encodings_from_file()

    if not encodeListKnown:
        images, classNames, classIDs = load_images_from_folder(PATH)

        # If face encodings are not saved, compute and save them
        encodeListKnown = findEncodings(images, classNames)
        save_encodings_to_file(encodeListKnown,classNames,classIDs)
        print('Encoding Complete')


    # Create and configure the Tkinter window
    window = Tk()
    window.title("Attendance")
    window.geometry('1400x800')

    # Create frames
    frame_home = Frame(window, bg='white')
    frame_home.place(relx=0, rely=0, relheight=1, relwidth=1)
    frame_home.grid_forget()  # Hide initially

    frame_second = Frame(window, bg='white')
    frame_second.place(relx=0, rely=0, relheight=1, relwidth=1)
    frame_second.grid_forget()  # Hide initially
    
    frame_login = Frame(window, bg='white')
    frame_login.place(relx=0, rely=0, relheight=1, relwidth=1)

    # Establish a database connection
    db_connection = get_database_connection()


    tableList = Listbox(frame_home, selectmode=tk.SINGLE)
    tableList.grid(row=0, column=1, columnspan=2, sticky=tk.E)
    tableList.config(width=100, height=30)
    # Create and configure the Listbox for displaying attended students
    theList = Listbox(frame_second, selectmode=tk.SINGLE)
    theList.grid(row=0, column=1, columnspan=2, sticky=tk.E)
    theList.config(width=100, height=30)

    # Create and configure the video frame label
    label = tk.Label(frame_second, width=600, height=600)
    label.grid(row=0, column=0, padx=10, pady=10)
    attended_label = tk.Label(frame_second, text="Attended: 0", font=('Arial', 12))
    attended_label.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)

    absent_label = tk.Label(frame_second, text="Absent: 0", font=('Arial', 12))
    absent_label.grid(row=1, column=0, padx=10, pady=5, sticky=tk.E)
    # Create buttons to switch between pages

    # Functions in frame_home
    switch_to_login(frame_home,frame_second)
    # Functions in frame_second

    # Display attended students data and start updating the video frame
    # window.protocol("WM_DELETE_WINDOW", release_capture)

    # Start the Tkinter event loop
    window.mainloop() 