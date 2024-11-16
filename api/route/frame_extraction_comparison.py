from flask import Blueprint, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import requests
from io import BytesIO
import base64
import json

# Blueprint for the frame extraction API
frame_extraction_api = Blueprint('frame_extraction_api', __name__)

# Define class names based on model training classes
class_names = [
    'hachijiDachi_jodanYoko', 
    'sanchinDachi_ageUke', 
    'sanchinDachi_jodanTsuki', 
    'sanchinDachi_sotoUke', 
    'shikoDachi_gedanBarai', 
    'sotoUke_maeGeri', 
    'zenkutsuDachi_awaseTsuki', 
    'zenkutsuDachi_chudanTsuki', 
    'zenkutsuDachi_empiUke'
]

# Load the model
model = tf.keras.models.load_model('models_files/Resnetmodel_1.keras')

# Define desired image dimensions and batch size
img_height, img_width = 384, 512
batch_size = 16
output_dir = 'output_best_frames'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Helper function to preprocess a frame
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (img_width, img_height))
    img_array = image.img_to_array(resized_frame)
    return img_array

# Function to classify and send frames in base64 to preprocessing API
def classify_and_send_frames(video_file):
    # Save the uploaded video temporarily
    temp_video_path = 'temp_video.mp4'
    video_file.save(temp_video_path)

    # Open video and set parameters
    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / 10)  # Extract frames at 10 FPS

    # Dictionary to store top predictions for each class
    class_predictions = {class_name: [] for class_name in class_names}
    detected_classes = set()
    frame_count, batch_frames = 0, []

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval)
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess and add to batch
        processed_frame = preprocess_frame(frame)
        batch_frames.append(processed_frame)

        if len(batch_frames) == batch_size:
            predictions = model.predict(np.array(batch_frames))
            for idx, prediction in enumerate(predictions):
                predicted_index = np.argmax(prediction)
                confidence = prediction[predicted_index]
                predicted_class_name = class_names[predicted_index]

                detected_classes.add(predicted_class_name)
                class_predictions[predicted_class_name].append((confidence, batch_frames[idx]))

                # Keep only top 5 frames per class by confidence
                class_predictions[predicted_class_name] = sorted(
                    class_predictions[predicted_class_name],
                    key=lambda x: x[0],
                    reverse=True
                )[:5]

            batch_frames.clear()  # Clear batch

        frame_count += 1

    # Release video and remove temp file
    cap.release()
    os.remove(temp_video_path)

    # Warn about undetected classes
    for class_name in class_names:
        if class_name not in detected_classes:
            print(f"Warning: Class '{class_name}' was not detected in the video.")

    # Prepare the JSON data with frames as base64-encoded strings
    frames_data = []
    for class_name, frames in class_predictions.items():
        for idx, (confidence, frame) in enumerate(frames):
            # Encode frame to JPEG
            is_success, buffer = cv2.imencode(".jpg", frame)
            if is_success:
                # Convert image to base64
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frames_data.append({
                    'class': class_name,
                    'confidence': confidence,
                    'frame_index': idx + 1,
                    'frame_data': frame_base64
                })

    # Send frames data as JSON to preprocessing API
    preprocessing_url = 'http://localhost:5000/preprocessing'
    payload = json.dumps({'frames': frames_data})
    headers = {'Content-Type': 'application/json'}

    response = requests.post(preprocessing_url, data=payload, headers=headers)

    return response.json() if response.status_code == 200 else {'error': 'Preprocessing failed'}

# Route to process local video file
@frame_extraction_api.route('/process_local_video', methods=['GET'])
def process_local_video():
    # Path to the local video file
    video_path = os.path.join(os.path.dirname(__file__), '..', 'video', 'kata.mp4')
    try:
        # Classify and send frames to preprocessing API
        response_data = classify_and_send_frames(video_path)

        # Check response from preprocessing API
        if 'error' in response_data:
            return jsonify({'error': response_data['error']}), 500

        # Return successful result
        return jsonify({
            'message': 'Frame extraction and preprocessing completed successfully',
            'preprocessing_result': response_data
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500














'''from flask import Blueprint, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

# Blueprint for the frame extraction API
frame_extraction_api = Blueprint('frame_extraction_api', __name__)

# Define class names based on model training classes
class_names = [
    'hachijiDachi_jodanYoko', 
    'sanchinDachi_ageUke', 
    'sanchinDachi_jodanTsuki', 
    'sanchinDachi_sotoUke', 
    'shikoDachi_gedanBarai', 
    'sotoUke_maeGeri', 
    'zenkutsuDachi_awaseTsuki', 
    'zenkutsuDachi_chudanTsuki', 
    'zenkutsuDachi_empiUke'
]

# Load the model
model = tf.keras.models.load_model('models_files/Resnetmodel_1.keras')

# Define desired image dimensions and batch size
img_height, img_width = 384, 512
batch_size = 16
output_dir = 'output_best_frames'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Helper function to preprocess a frame
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (img_width, img_height))
    img_array = image.img_to_array(resized_frame)
    return img_array

# Function to classify and send frames
def classify_and_send_frames(video_file):
    # Save the uploaded video temporarily
    temp_video_path = 'temp_video.mp4'
    video_file.save(temp_video_path)

    # Open video and set parameters
    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / 10)  # Extract frames at 10 FPS

    # Dictionary to store top predictions for each class
    class_predictions = {class_name: [] for class_name in class_names}
    detected_classes = set()
    frame_count, batch_frames = 0, []

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval)
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess and add to batch
        processed_frame = preprocess_frame(frame)
        batch_frames.append(processed_frame)

        if len(batch_frames) == batch_size:
            predictions = model.predict(np.array(batch_frames))
            for idx, prediction in enumerate(predictions):
                predicted_index = np.argmax(prediction)
                confidence = prediction[predicted_index]
                predicted_class_name = class_names[predicted_index]

                detected_classes.add(predicted_class_name)
                class_predictions[predicted_class_name].append((confidence, batch_frames[idx]))

                # Keep only top 5 frames per class by confidence
                class_predictions[predicted_class_name] = sorted(
                    class_predictions[predicted_class_name],
                    key=lambda x: x[0],
                    reverse=True
                )[:5]

            batch_frames.clear()  # Clear batch

        frame_count += 1

    # Release video and remove temp file
    cap.release()
    os.remove(temp_video_path)

    # Warn about undetected classes
    for class_name in class_names:
        if class_name not in detected_classes:
            print(f"Warning: Class '{class_name}' was not detected in the video.")
    
    # Prepare and send images as files to preprocessing API
    with requests.Session() as session:
        files = []
        for class_name, frames in class_predictions.items():
            for idx, (confidence, frame) in enumerate(frames):
                # Encode each frame as JPEG in memory
                is_success, buffer = cv2.imencode(".jpg", frame)
                if is_success:
                    file_data = BytesIO(buffer.tobytes())
                    file_data.name = f"{class_name}_{idx + 1}.jpg"
                    files.append(('frames', (file_data.name, file_data, 'image/jpeg')))

        # Send files to preprocessing API
        preprocessing_url = 'http://localhost:5000/preprocessing'
        response = session.post(preprocessing_url, files=files)

    return response.json() if response.status_code == 200 else {'error': 'Preprocessing failed'}

# Route to extract and classify frames
@frame_extraction_api.route('/frame_extraction_comparison', methods=['POST'])
def frame_extraction_comparison():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    try:
        # Classify and send frames to preprocessing API
        response_data = classify_and_send_frames(video_file)

        # Check response from preprocessing API
        if 'error' in response_data:
            return jsonify({'error': response_data['error']}), 500

        # Return successful result
        return jsonify({
            'message': 'Frame extraction and preprocessing completed successfully',
            'preprocessing_result': response_data
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500'''
