# AI Object Detection

An AI-based object detection application built using Python, Streamlit, OpenCV, NumPy, Pandas, and YOLO.

## Project Description

This project detects objects in images, webcam input, and video using a YOLO object detection model.

The application provides a simple Streamlit interface where users can select the input type and view the detected objects with their confidence scores.

## Technologies Used

- Python
- Streamlit
- YOLO (Ultralytics)
- OpenCV
- NumPy
- Pandas
- Pillow

## Features

- Image object detection
- Real-time webcam object detection
- Video object detection
- Object bounding boxes
- Confidence scores
- Simple Streamlit user interface

## Model

The project currently uses the `yolo11s.pt` YOLO model.

The current pretrained model is mainly being used for general object detection. Custom object detection classes such as pen detection may require a separately trained model.

## Project Structure

```text
AI_Object_Detection/
│
├── app.py
├── requirements.txt
├── yolo11s.pt
├── .gitignore
└── README.md