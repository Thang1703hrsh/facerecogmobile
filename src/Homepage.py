import streamlit as st
import tensorflow as tf
import re
import argparse
import facenet
import imutils
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import tempfile
from PIL import Image
from st_pages import Page, show_pages, add_page_title
import time

st.set_page_config(page_title="Face Recognition App", layout="wide")
add_page_title()

show_pages(
    [
        Page("Homepage.py", "Home", "🏠"),
        Page("pages/Updating.py", "Updating", "🔄"),
        Page("pages/Database.py", "Database", "📊"),
    ]
)

# Sidebar - Settings
st.sidebar.title("Settings")

st.sidebar.subheader("Recognition Tolerance")
TOLERANCE = st.sidebar.slider("Tolerance", 0.0, 1.0, 0.5, 0.01)
st.sidebar.info("Lower tolerance is stricter, higher tolerance is looser for face recognition.")

# Common Settings for both Video, Webcam, and Images
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = '../Models/facemodel.pkl'
FACENET_MODEL_PATH = '../Models/20180402-114759.pb'
detection_time_placeholder = st.sidebar.empty()
model_name = "Facenet"

# Load The Custom Classifier
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)

# Load Feature Extraction Model
facenet.load_model(FACENET_MODEL_PATH)

# Initialize TensorFlow session and GPU settings
tf.Graph().as_default()
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

# Get input/output tensors
images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "align")

# Start recognition button
start_recognition = st.button("Start Recognition")
stop_recognition = st.button("Stop Recognition")

# Initialize webcam only when "Start Recognition" is clicked
if start_recognition:
    if "cap" not in st.session_state:
        st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not st.session_state.cap.isOpened():
            st.error("Failed to open webcam!")
        else:
            FRAME_WINDOW = st.empty()

    # Webcam frame capture loop
    while True:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        frame = imutils.resize(frame, width=1200, height=600)
        frame = cv2.flip(frame, 1)

        start_time = time.time()

        # Detect faces
        bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

        faces_found = bounding_boxes.shape[0]
        if faces_found > 0:
            for det in bounding_boxes:
                bb = det.astype(int)
                
                cropped = frame[bb[1]:bb[3], bb[0]:bb[2]]
                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
                scaled = facenet.prewhiten(scaled).reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

                # Make predictions
                feed_dict = {images_placeholder: scaled, phase_train_placeholder: False}
                emb_array = sess.run(embeddings, feed_dict=feed_dict)
                predictions = model.predict_proba(emb_array)
                best_class_idx = np.argmax(predictions)
                best_prob = predictions[0, best_class_idx]
                name = class_names[best_class_idx] if best_prob > TOLERANCE else "Unknown"

                # Draw bounding box and label
                cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({best_prob:.2f})", (bb[0], bb[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Stop recognition button functionality
if stop_recognition:
    if "cap" in st.session_state:
        st.session_state.cap.release()  # Release the webcam
        cv2.destroyAllWindows()  # Close any OpenCV windows
        st.success("Webcam stopped.")

