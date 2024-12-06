import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import cv2
from src import facenet
from src.align import detect_face
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av 

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

# Callback for processing webcam frames
class VideoProcessor:
    def recv(self, frame):
        # Convert the stream's image to a numpy array (OpenCV format)
        img = frame.to_ndarray(format="bgr24")
        
        # Detect faces
        bounding_boxes, _ = detect_face.detect_face(img, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

        faces_found = bounding_boxes.shape[0]
        if faces_found > 0:
            for det in bounding_boxes:
                bb = det.astype(int)

                # Crop and scale the detected face
                cropped = img[bb[1]:bb[3], bb[0]:bb[2]]
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
                cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                cv2.putText(img, f"{name} ({best_prob:.2f})", (bb[0], bb[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Return the processed frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")
        st.image(processed_image, caption='Detected Video', channels="BGR", use_column_width=True)

rtc_configuration={  # Add this config
        "iceServers": [
      {
        "urls": "stun:stun.relay.metered.ca:80",
      },
      {
        "urls": "turn:global.relay.metered.ca:80",
        "username": "bbfcabc563f99fa725cc4668",
        "credential": "w2khlKKUp+qjdsS0",
      },
      {
        "urls": "turn:global.relay.metered.ca:80?transport=tcp",
        "username": "bbfcabc563f99fa725cc4668",
        "credential": "w2khlKKUp+qjdsS0",
      },
      {
        "urls": "turn:global.relay.metered.ca:443",
        "username": "bbfcabc563f99fa725cc4668",
        "credential": "w2khlKKUp+qjdsS0",
      },
      {
        "urls": "turns:global.relay.metered.ca:443?transport=tcp",
        "username": "bbfcabc563f99fa725cc4668",
        "credential": "w2khlKKUp+qjdsS0",
      },
  ],
}

# Set up the WebRTC streamer for the video feed
webrtc_streamer(key="face-recognition", mode=WebRtcMode.SENDRECV, video_processor_factory=VideoProcessor, 
                rtc_configuration= rtc_configuration)
