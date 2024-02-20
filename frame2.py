import os
import cv2
from scipy import ndimage
import glob
import numpy as np

def enhanceImage(image):
    # Non-Linear filter for noise removal
    deNoised = cv2.medianBlur(image, 3)

    # Convert to LAB color space for histogram equalization
    lab = cv2.cvtColor(deNoised, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE-enhanced L channel with the original A and B channels
    limg = cv2.merge((cl, a, b))

    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Sharpening filter to enhance details
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]], dtype=np.float32)
    enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)

    return enhanced_image

def frame_video(video_file, frame_folder):
    # Ensure frame folder exists
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    video_frame_folder = os.path.join(frame_folder, video_name)

    # Ensure video-specific frame folder exists
    if not os.path.exists(video_frame_folder):
        os.makedirs(video_frame_folder)

    print(f"Processing {video_file}...")

    vidcap = cv2.VideoCapture(video_file)

    if not vidcap.isOpened():
        print(f"Failed to open {video_file}")
        return

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # Interval to capture frame (every second)
    count = 0
    frame_count = 0

    while vidcap.isOpened():
        success, image = vidcap.read()
        if not success:
            break

        if count % frame_interval == 0:
            # Enhance image before saving
            enhanced_image = enhanceImage(image)

            frame_name = os.path.join(video_frame_folder, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_name, enhanced_image)  # Save frame as JPEG file
            print(f"Saved {frame_name}")
            frame_count += 1

        count += 1

    vidcap.release()

# Usage


