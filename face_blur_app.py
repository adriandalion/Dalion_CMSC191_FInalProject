import cv2
import streamlit as st
import numpy as np
import datetime
import os

# Load Haar cascade for face detection using OpenCV's pre-trained classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up Streamlit page configuration and title
st.set_page_config(page_title="Face Blurring and Anonymization App", layout="centered")
st.title("🕵️ Real-Time Face Blurring")

# Display application description and features
st.markdown("""
This app uses your webcam to detect and blur faces in real time using **OpenCV Haar Cascades** with enhanced filtering and capture functionality.

**Features:**
- Real-time webcam capture
- Improved face detection accuracy
- Blur or blackout anonymization options
- Face labeling with total count
- Screenshot capture button
""")

# Sidebar settings for mode selection, sensitivity, screenshot, and camera toggle
st.sidebar.title("Settings")
anonymization_mode = st.sidebar.radio("Choose anonymization mode:", ["Blur Faces", "Black Out Faces"])
detection_sensitivity = st.sidebar.slider("Detection Sensitivity (higher = stricter)", 3, 10, 6)
capture_btn = st.sidebar.button("📸 Capture Screenshot")
runtime_toggle = st.sidebar.checkbox('✅ Start Webcam')

# Containers for displaying image and status in the main UI
stframe = st.empty()
status_text = st.empty()

# Start video capture loop if webcam checkbox is enabled
if runtime_toggle:
    cap = cv2.VideoCapture(0)  # Start webcam feed
    if not cap.isOpened():
        st.error("❌ Cannot access webcam. Please make sure it's connected.")  # Error if webcam is inaccessible
    else:
        screenshot_captured = False  # Flag to prevent duplicate captures in one click
        while True:
            ret, frame = cap.read()  # Read frame from webcam
            if not ret:
                st.warning("⚠️ Failed to capture frame from webcam.")  # Show warning if frame is not read properly
                break

            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces with adjustable sensitivity
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=detection_sensitivity,
                minSize=(100, 100),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            face_count = 0  # Counter for detected faces

            # Loop through each detected face
            for (x, y, w, h) in faces:
                face_count += 1
                roi = frame[y:y+h, x:x+w]  # Region of interest (face area)

                # Apply blur or blackout based on selected mode
                if anonymization_mode == "Blur Faces":
                    blur = cv2.GaussianBlur(roi, (51, 51), 30)
                    frame[y:y+h, x:x+w] = blur
                else:
                    frame[y:y+h, x:x+w] = 0  # Black out the face region

                # Draw rectangle and label on each detected face
                label = f"Face #{face_count}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Display total number of faces detected on screen
            cv2.putText(frame, f"Total Faces: {face_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Handle screenshot capture if button is clicked
            if capture_btn and not screenshot_captured:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
                cv2.imwrite(filename, frame)  # Save the current frame
                st.sidebar.success(f"Screenshot saved as {filename}")  # Notify user
                screenshot_captured = True
            elif not capture_btn:
                screenshot_captured = False  # Reset flag when button is released

            # Convert frame to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the processed frame and update status
            stframe.image(frame_rgb, channels="RGB")
            status_text.info(f"📸 Live feed active. {face_count} face(s) detected.")

        # Release camera and close OpenCV windows when done
        cap.release()
        cv2.destroyAllWindows()
