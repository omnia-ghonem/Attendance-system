# Face Recognition Attendance System

## Overview

The Face Recognition Attendance System is an innovative project aimed at automating the attendance recording process using facial recognition technology. This system streamlines the attendance management process in various settings, making it more efficient and accurate.

## Features

- **Facial Recognition:** Automatically identifies individuals using facial recognition algorithms.
- **Attendance Recording:** Records the attendance with the name, date, and time of the individual.
- **Image Capture & Upload:** Allows image capture via webcam or the upload of images for recognition.
- **Web Interface:** A user-friendly web interface for interacting with the system and displaying attendance records.
- **Export Attendance Data:** The attendance data can be exported as a CSV file for easy handling and storage.

## Technology Stack

- **Programming Language:** Python
- **Web Framework:** Flask (for web requests and serving the application)
- **Libraries:** 
  - OpenCV (for image processing and face detection)
  - dlib (for face recognition)
  - Flask (for the web interface)
- **Data Handling:** 
  - CSV files for storing attendance data
  - `encodings.pkl` for storing face recognition data

## Components

1. **app.py:** 
   - The central Flask web application that handles web requests, routes, and integrates facial recognition functionality.
   
2. **Attendance.csv:**
   - A CSV file that stores the attendance records, including names, dates, and times of attendees.

3. **Face Recognition Components:**
   - `predict.py`, `train.py`, and other scripts handle face detection and recognition.
   - Model data stored in `encodings.pkl` and `face_encodings.pickle`.

4. **Utility Scripts:**
   - Manage database entries and enhance recognition processes for improved accuracy.

5. **Web Interface:**
   - HTML, CSS, and static files used to create a user-friendly web interface.
   
6. **Image Data:**
   - The `uploads` folder stores uploaded images, and the `Image_Attendance` folder is used for storing related images for attendance records.

7. **Additional Scripts:**
   - Scripts such as `vision.py` and `frame2.py` contribute to different aspects of the system and help with continuous development.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/omnia-ghonem/Attendance-system.git
   ```

2. **Install dependencies:**
   Make sure you have Python 3.x installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   Start the Flask web application:
   ```bash
   python app.py
   ```

4. **Access the system:**
   Open your browser and go to `http://127.0.0.1:5000/` to interact with the attendance system.

## Data Privacy

The system takes privacy concerns into account when handling biometric data. All facial recognition data is stored securely, and the system ensures efficient processing with minimal data exposure.

## Conclusion

This project leverages cutting-edge facial recognition technology and web development tools to automate attendance management, improving operational efficiency and accuracy in various environments.
