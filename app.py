from flask import Flask, render_template, Response, request, jsonify
import tensorflow as tf
import pickle
import imutils
import cv2
import numpy as np
import time
from src import facenet
from src.align import detect_face

# Initialize Flask app
app = Flask(__name__)

# Common Settings for both Video, Webcam, and Images
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'Models/facemodel.pkl'
FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

# Load the custom classifier
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)

# Load feature extraction model
facenet.load_model(FACENET_MODEL_PATH)

# Initialize TensorFlow session and GPU settings
tf.compat.v1.Session()
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

# Get input/output tensors
images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
pnet, rnet, onet = detect_face.create_mtcnn(sess, "src/align")

# Video capture object
cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and resize the frame
        frame = imutils.resize(frame, width=1200, height=600)
        frame = cv2.flip(frame, 1)

        # Detect faces
        bounding_boxes, _ = detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
        
        faces_found = bounding_boxes.shape[0]
        if faces_found > 0:
            for det in bounding_boxes:
                bb = det.astype(int)
                
                # Crop, scale, and predict
                cropped = frame[bb[1]:bb[3], bb[0]:bb[2]]
                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
                scaled = facenet.prewhiten(scaled).reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                feed_dict = {images_placeholder: scaled, phase_train_placeholder: False}
                emb_array = sess.run(embeddings, feed_dict=feed_dict)
                predictions = model.predict_proba(emb_array)
                best_class_idx = np.argmax(predictions)
                best_prob = predictions[0, best_class_idx]
                name = class_names[best_class_idx] if best_prob > 0.5 else "Unknown"

                # Draw bounding box and label
                cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({best_prob:.2f})", (bb[0], bb[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    # Start recognition (turn on webcam)
    global cap
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return jsonify({"status": "Recognition started"})

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    # Stop recognition (release webcam)
    global cap
    if cap.isOpened():
        cap.release()
        cv2.destroyAllWindows()
    return jsonify({"status": "Recognition stopped"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
