from flask import Blueprint, jsonify
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


# Blueprint for the frame extraction API
frame_extraction_api = Blueprint('frame_extraction_api',__name__)

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
model = tf.keras.models.load_model('models_files/Resnetmodel_final.keras')

# Define desired image dimensions and batch size
img_height, img_width = 384, 512
batch_size = 16

"""def preprocess_frame(frame):
    try:
        resized_frame = cv2.resize(frame, (img_width, img_height))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img_array = image.img_to_array(rgb_frame) # Normalize pixel values
        return img_array
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return None"""

def preprocess_frame(frames):
    preprocessed_frames = []
    for frame in frames:
        try:
            if isinstance(frame, str) and os.path.exists(frame):  # Handle file paths
                img = tf.keras.preprocessing.image.load_img(frame, target_size=(384, 512))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
            elif isinstance(frame, dict) and 'frame_data' in frame:  # Handle base64-encoded frames
                img_data = base64.b64decode(frame['frame_data'])
                img = Image.open(io.BytesIO(img_data)).resize((512, 384))  # Resizing here
                img_array = tf.keras.preprocessing.image.img_to_array(img)
            elif isinstance(frame, np.ndarray):  # Handle already preprocessed frames
                # Resize using OpenCV for numpy arrays
                img_array = cv2.resize(frame, (512, 384))
            else:
                raise ValueError(f"Unexpected frame format: {type(frame)}")

            preprocessed_frames.append(img_array)
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

    return np.array(preprocessed_frames)

    
def classify_and_send_frames(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        return {'error': f"Failed to open video file: {video_file_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 FPS if not available
    frame_interval = max(int(fps / 10), 1)  # Minimum interval is 1

    class_predictions = {class_name: [] for class_name in class_names}
    detected_classes = set()
    batch_frames, frame_count = [], 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            processed_frame = preprocess_frame(frame)
            if processed_frame is not None:
                batch_frames.append(processed_frame)

            if len(batch_frames) == batch_size:
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

                batch_frames.clear()

        frame_count += 1

    cap.release()

    frames_data = []
    for class_name, frames in class_predictions.items():
        for idx, (confidence, frame) in enumerate(frames):
            is_success, buffer = cv2.imencode(".jpg", (frame * 255).astype(np.uint8))
            if is_success:
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frames_data.append({
                    'class': class_name,
                    'confidence': float(confidence),
                    'frame_index': idx + 1,
                    'frame_data': frame_base64
                })

    preprocessing_url = 'http://localhost:5000/api/preprocessing_api/preprocessing'

    payload = json.dumps({'frames': frames_data})
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(preprocessing_url, data=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': f"Preprocessing API request failed: {e}"}

@frame_extraction_api.route('/process_local_video', methods=['GET'])
def process_local_video():
    video_path = r'D:\UpWork\NewRepo\video\newV6.mp4'

    if not os.path.exists(video_path):
        return jsonify({'error': f"Video file not found at {video_path}"}), 404

    try:
        response_data = classify_and_send_frames(video_path)
        if 'error' in response_data:
            return jsonify({'error': response_data['error'], 'details': response_data.get('details', '')}), 400

        return jsonify({
            'message': 'Frame extraction and preprocessing completed successfully',
            'preprocessing_result': response_data
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
