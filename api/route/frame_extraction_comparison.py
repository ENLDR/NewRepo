from flask import Blueprint, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import requests
import base64
import json
import io
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

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
logging.debug("Loading model...")
try:
    model = tf.keras.models.load_model('models_files/Resnetmodel_final.keras')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Define desired image dimensions and batch size
img_height, img_width = 384, 512
batch_size = 16

def preprocess_frame(frame):
    try:
        logging.debug("Preprocessing a frame...")
        resized_frame = cv2.resize(frame, (img_width, img_height))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img_array = image.img_to_array(rgb_frame)  # Normalize pixel values
        return img_array
    except Exception as e:
        logging.error(f"Error preprocessing frame: {e}")
        return None

def classify_and_send_frames(video_file_path,email,video_name):
    logging.info(f"Processing video: {video_file_path}")
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        error_msg = f"Failed to open video file: {video_file_path}"
        logging.error(error_msg)
        return {'error': error_msg}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(int(fps / 10), 1)
    logging.debug(f"Video FPS: {fps}, Frame interval: {frame_interval}")

    class_predictions = {class_name: [] for class_name in class_names}
    detected_classes = set()
    batch_frames, frame_count = [], 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("Reached end of video.")
            break

        if frame_count % frame_interval == 0:
            logging.debug(f"Processing frame {frame_count}...")
            processed_frame = preprocess_frame(frame)
            if processed_frame is not None:
                batch_frames.append(processed_frame)

            if len(batch_frames) == batch_size:
                try:
                    logging.debug("Predicting batch of frames...")
                    predictions = model.predict(np.array(batch_frames))
                    for idx, prediction in enumerate(predictions):
                        predicted_index = np.argmax(prediction)
                        confidence = prediction[predicted_index]
                        predicted_class_name = class_names[predicted_index]

                        detected_classes.add(predicted_class_name)
                        class_predictions[predicted_class_name].append((confidence, batch_frames[idx]))

                        # Keep only the top 5 frames per class
                        class_predictions[predicted_class_name] = sorted(
                            class_predictions[predicted_class_name],
                            key=lambda x: x[0],  # Sort by confidence
                            reverse=True
                        )[:5]
                except Exception as e:
                    logging.error(f"Error during prediction: {e}")
                batch_frames.clear()

        frame_count += 1

    cap.release()

    frames_data = []
    for class_name, frames in class_predictions.items():
        for _, frame in frames:
            is_success, buffer = cv2.imencode(".jpg", (frame * 255).astype(np.uint8))
            if is_success:
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frames_data.append(frame_base64)

    logging.info("Sending frames to preprocessing API.")
    response = send_frames_to_preprocessing(frames_data,email,video_name)
    return response

def send_frames_to_preprocessing(frames_data,email,video_name):
    preprocessing_url = 'http://10.95.147.34:5000/api/preprocessing_api/preprocessing'
    payload = {'frames': frames_data,'email': email,'video_name':video_name}
    headers = {'Content-Type': 'application/json'}
    logging.debug("Sending frames to preprocessing API...")

    
    try:
        response = requests.post(preprocessing_url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()
        logging.debug("Received response from preprocessing API.")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Preprocessing API request failed: {e}")
        return {'error': f"Preprocessing API request failed: {e}"}
    

@frame_extraction_api.route('/process_local_video', methods=['POST'])
def process_local_video():

    email = request.get_json().get('email')
    _title = request.get_json().get('title')
    logging.info(email)
    video_path = f"D:/UpWork/NewRepo/uploads/{_title}"

    if not os.path.exists(video_path):
        error_msg = f"Video file not found at {video_path}"
        logging.error(error_msg)
        return jsonify({'error': error_msg}), 404

    try:
        
        response_data = classify_and_send_frames(video_path,email,_title)
        
        if 'error' in response_data:
            return jsonify({'error': response_data['error']}), 401

        return jsonify({
            'message': 'Frame extraction and preprocessing completed successfully',
            'preprocessing_result': response_data
        }), 200
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500