# Dalion_CMSC191_FInalProject

Real-Time Face Blurring and Anonymization App

A Python-based web application that uses OpenCV and Streamlit to perform real-time face detection and anonymization via your webcam. The app offers two anonymization modes—Blur and Black Out—along with live face counting, labeling, sensitivity adjustment, and screenshot capture features.

------------------------------------------------------------

Features

- Real-time face detection using OpenCV Haar Cascades
- Adjustable detection sensitivity via Streamlit slider
- Two anonymization modes: Blur and Black Out
- Screenshot capture with automatic timestamped filename
- Live face count and labeling (e.g., Face #1, Face #2)
- Clean and interactive Streamlit interface

------------------------------------------------------------

Requirements

- Python 3.7–3.11
- OpenCV
- Streamlit
- NumPy

Install requirements with:
pip install streamlit opencv-python numpy

------------------------------------------------------------

How to Run

1. Clone or download this repository
2. Open your terminal or command prompt in the project folder
3. Run the Streamlit app:
   streamlit run face_blur_haar_accurate.py
4. The app will open in your default browser

------------------------------------------------------------

How to Use

- Start Webcam – Check this to activate your webcam
- Choose Mode – Select between "Blur Faces" or "Black Out Faces"
- Adjust Sensitivity – Use the slider for more accurate detection (higher = stricter)
- Capture Screenshot – Saves the current frame as a PNG file with a timestamp

------------------------------------------------------------

Screenshot Storage

Screenshots are saved in the same folder as the Python script:
screenshot_YYYYMMDD_HHMMSS.png

------------------------------------------------------------

Notes

- Only detects frontal faces using Haar cascades
- Accuracy depends on lighting and camera quality
- Increase sensitivity to improve multi-face detection

------------------------------------------------------------

Example Use Cases

- Privacy masking in video calls
- Educational demos on computer vision
- Prototype for surveillance or streaming filters

------------------------------------------------------------

References

- OpenCV: https://opencv.org/
- Streamlit: https://streamlit.io/
- Haar Cascades: https://github.com/opencv/opencv/tree/master/data/haarcascades

------------------------------------------------------------
