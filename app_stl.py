import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import cv2
from src import facenet
from src.align import detect_face
from PIL import Image

# Streamlit page configuration
st.set_page_config(page_title="Face Recognition App", layout="wide")
st.title("Welcome to facial recognition")

# Sidebar Settings
st.sidebar.title("Settings")
TOLERANCE = st.sidebar.slider("Recognition Tolerance", 0.0, 1.0, 0.5, 0.01)
st.sidebar.info("Lower tolerance is stricter, higher tolerance is looser for face recognition.")

# Common Settings
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'facemodel.pkl'
FACENET_MODEL_PATH = '20180402-114759.pb'

# Load Classifier Model
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)

# Load Feature Extraction Model only once
st.write("Loading FaceNet Model...")
facenet.load_model(FACENET_MODEL_PATH)

# Initialize TensorFlow session and GPU settings only once
tf.Graph().as_default()
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

# Get input/output tensors
images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
pnet, rnet, onet = detect_face.create_mtcnn(sess, "src/align")

# Image upload functionality
st.title("Face Recognition App")
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Start recognition button
if uploaded_image is not None:
    # Convert the uploaded file to a NumPy array and load into an image
    image = Image.open(uploaded_image)
    image = np.array(image)

    # Detect faces
    bounding_boxes, _ = detect_face.detect_face(image, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

    faces_found = bounding_boxes.shape[0]
    if faces_found > 0:
        for det in bounding_boxes:
            bb = det.astype(int)
            
            # Crop and resize the face
            cropped = image[bb[1]:bb[3], bb[0]:bb[2]]
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
            cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
            cv2.putText(image, f"{name} ({best_prob:.2f})", (bb[0], bb[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        st.image(image, caption="Processed Image", use_column_width=True)
    else:
        st.error("No faces detected in the image.")