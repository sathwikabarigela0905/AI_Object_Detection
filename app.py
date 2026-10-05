import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from dotenv import load_dotenv
from ultralytics import YOLO


# ============================================================
# LOAD SETTINGS
# ============================================================

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "yolo11n.pt")

DEFAULT_CONFIDENCE = float(
    os.getenv("DEFAULT_CONFIDENCE", "0.50")
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Based Object Detection Platform",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Based Object Detection Platform")

st.write(
    "AI-powered object detection using YOLO and Computer Vision."
)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_NAME)


try:
    model = load_model()

except Exception as e:
    st.error(f"Unable to load YOLO model: {e}")
    st.stop()


# ============================================================
# DETECTION SETTINGS
# ============================================================

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.95,
    value=DEFAULT_CONFIDENCE,
    step=0.05
)


# ============================================================
# OBJECT DETECTION FUNCTION
# ============================================================

def detect_objects(image):

    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    annotated_image = result.plot()

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            score = float(box.conf[0])

            detections.append({
                "Object": class_name,
                "Confidence": score
            })

    return annotated_image, detections


# ============================================================
# STATISTICS FUNCTION
# ============================================================

def show_statistics(detections):

    if not detections:

        st.warning("No objects detected.")

        return

    df = pd.DataFrame(detections)

    total_objects = len(df)

    object_types = df["Object"].nunique()

    average_confidence = df["Confidence"].mean()

    highest_confidence = df["Confidence"].max()


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Objects",
        total_objects
    )

    col2.metric(
        "Object Types",
        object_types
    )

    col3.metric(
        "Average Confidence",
        f"{average_confidence:.1%}"
    )

    col4.metric(
        "Highest Confidence",
        f"{highest_confidence:.1%}"
    )


    # --------------------------------------------------------
    # OBJECT COUNT
    # --------------------------------------------------------

    st.subheader("Object Count")

    counts = (
        df["Object"]
        .value_counts()
        .rename_axis("Object")
        .reset_index(name="Count")
    )

    st.dataframe(
        counts,
        width="stretch",
        hide_index=True
    )


    # --------------------------------------------------------
    # OBJECT COUNT CHART
    # --------------------------------------------------------

    chart_data = counts.set_index("Object")

    st.bar_chart(
        chart_data,
        width="stretch"
    )


    # --------------------------------------------------------
    # DETECTION DETAILS
    # --------------------------------------------------------

    st.subheader("Detection Details")

    display_df = df.copy()

    display_df["Confidence"] = (
        display_df["Confidence"] * 100
    ).round(2).astype(str) + "%"

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True
    )


# ============================================================
# TABS
# ============================================================

image_tab, webcam_tab, video_tab = st.tabs(
    [
        "Image Detection",
        "Webcam",
        "Video Detection"
    ]
)


# ============================================================
# IMAGE DETECTION
# ============================================================

with image_tab:

    st.header("Image Object Detection")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "jfif",
            "png",
            "bmp",
            "webp"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_array = np.array(image)


        # ----------------------------------------------------
        # ORIGINAL + RESULT
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            st.subheader("Original Image")

            st.image(
                image,
                width="stretch"
            )


        with col2:

            st.subheader("Detection Result")

            if st.button(
                "Detect Objects",
                type="primary"
            ):

                with st.spinner(
                    "Detecting objects..."
                ):

                    annotated_image, detections = (
                        detect_objects(
                            image_array
                        )
                    )


                st.image(
                    annotated_image,
                    channels="BGR",
                    width="stretch"
                )


                st.session_state["image_result"] = (
                    annotated_image
                )

                st.session_state["image_detections"] = (
                    detections
                )


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        if "image_detections" in st.session_state:

            st.divider()

            show_statistics(
                st.session_state["image_detections"]
            )


            # ------------------------------------------------
            # DOWNLOAD IMAGE
            # ------------------------------------------------

            result = st.session_state["image_result"]

            result_rgb = cv2.cvtColor(
                result,
                cv2.COLOR_BGR2RGB
            )

            success, encoded = cv2.imencode(
                ".jpg",
                result_rgb
            )

            if success:

                st.download_button(
                    "Download Detected Image",
                    encoded.tobytes(),
                    "detected_image.jpg",
                    "image/jpeg"
                )


# ============================================================
# WEBCAM DETECTION
# ============================================================

with webcam_tab:

    st.header("Webcam Object Detection")

    camera_image = st.camera_input(
        "Take a picture"
    )


    if camera_image is not None:

        image = Image.open(
            camera_image
        ).convert("RGB")

        image_array = np.array(image)


        with st.spinner(
            "Detecting objects..."
        ):

            annotated_image, detections = (
                detect_objects(
                    image_array
                )
            )


        st.subheader("Detection Result")


        st.image(
            annotated_image,
            channels="BGR",
            width="stretch"
        )


        st.divider()


        show_statistics(
            detections
        )


# ============================================================
# VIDEO DETECTION
# ============================================================

with video_tab:

    st.header("Video Object Detection")


    uploaded_video = st.file_uploader(
        "Upload a video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ],
        key="video_upload"
    )


    if uploaded_video is not None:

        st.video(uploaded_video)


        if st.button(
            "Start Video Detection",
            type="primary"
        ):

            input_path = "input_video.mp4"

            output_path = "output_video.mp4"


            # ------------------------------------------------
            # SAVE INPUT VIDEO
            # ------------------------------------------------

            with open(
                input_path,
                "wb"
            ) as file:

                file.write(
                    uploaded_video.getbuffer()
                )


            # ------------------------------------------------
            # OPEN VIDEO
            # ------------------------------------------------

            cap = cv2.VideoCapture(
                input_path
            )


            if not cap.isOpened():

                st.error(
                    "Unable to open the video."
                )


            else:

                fps = cap.get(
                    cv2.CAP_PROP_FPS
                )


                if fps <= 0:

                    fps = 25.0


                width = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                )


                height = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                )


                total_frames = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )


                # ------------------------------------------------
                # VIDEO WRITER
                # ------------------------------------------------

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )


                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps,
                    (width, height)
                )


                progress = st.progress(0)

                status = st.empty()


                frame_count = 0

                all_objects = []


                # ------------------------------------------------
                # PROCESS VIDEO
                # ------------------------------------------------

                while True:

                    success, frame = cap.read()


                    if not success:

                        break


                    results = model.predict(
                        frame,
                        conf=confidence,
                        verbose=False
                    )


                    result = results[0]


                    annotated_frame = result.plot()


                    writer.write(
                        annotated_frame
                    )


                    # --------------------------------------------
                    # COLLECT DETECTED OBJECTS
                    # --------------------------------------------

                    if result.boxes is not None:

                        for box in result.boxes:

                            class_id = int(
                                box.cls[0]
                            )


                            class_name = (
                                model.names[class_id]
                            )


                            all_objects.append(
                                class_name
                            )


                    frame_count += 1


                    # --------------------------------------------
                    # PROGRESS
                    # --------------------------------------------

                    if total_frames > 0:

                        progress.progress(
                            min(
                                frame_count / total_frames,
                                1.0
                            )
                        )


                    status.write(
                        f"Processing frame "
                        f"{frame_count}/{total_frames}"
                    )


                # ------------------------------------------------
                # RELEASE VIDEO
                # ------------------------------------------------

                cap.release()

                writer.release()


                progress.progress(1.0)


                status.success(
                    "Video detection completed."
                )


                # ------------------------------------------------
                # READ OUTPUT VIDEO
                # ------------------------------------------------

                with open(
                    output_path,
                    "rb"
                ) as video_file:

                    video_bytes = video_file.read()


                st.subheader(
                    "Processed Video"
                )


                st.video(
                    video_bytes
                )


                # ------------------------------------------------
                # DOWNLOAD VIDEO
                # ------------------------------------------------

                st.download_button(
                    "Download Processed Video",
                    video_bytes,
                    "detected_video.mp4",
                    "video/mp4"
                )


                # ------------------------------------------------
                # VIDEO SUMMARY
                # ------------------------------------------------

                if all_objects:

                    st.subheader(
                        "Video Detection Summary"
                    )


                    object_counts = (
                        pd.Series(all_objects)
                        .value_counts()
                        .rename_axis("Object")
                        .reset_index(
                            name="Detections"
                        )
                    )


                    st.dataframe(
                        object_counts,
                        width="stretch",
                        hide_index=True
                    )


                    # --------------------------------------------
                    # VIDEO CHART
                    # --------------------------------------------

                    video_chart_data = (
                        object_counts.set_index(
                            "Object"
                        )
                    )


                    st.bar_chart(
                        video_chart_data,
                        width="stretch"
                    )


                else:

                    st.warning(
                        "No objects were detected."
                    )