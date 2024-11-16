# preprocessing.py
from flask import Blueprint, jsonify, request
import requests
import tensorflow as tf
import numpy as np  # Import numpy for array handling

preprocessing_api = Blueprint('preprocessing_api', __name__)

# Load your model globally
model_path = 'C:/path/to/your/model.keras'  # Change this to your model path
model = tf.keras.models.load_model(model_path)

def preprocess_frames(frame_paths):
    """
    Preprocess the frames by loading images, resizing, and normalizing.
    Args:
        frame_paths (list): List of paths to the frames.
    Returns:
        np.array: Array of preprocessed frames.
    """
    preprocessed_frames = []

    for frame_path in frame_paths:
        # Load the image from frame_path, preprocess it
        image = tf.keras.preprocessing.image.load_img(frame_path, target_size=(384, 512))  # Adjust dimensions as needed
        image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # Normalize the image
        preprocessed_frames.append(image_array)

    # Convert the list of preprocessed frames into a NumPy array
    return np.array(preprocessed_frames)

@preprocessing_api.route('/preprocessing', methods=['POST'])
def preprocessing():
    data = request.json
    frames = data.get('frames')

    if not frames:
        return jsonify({'error': 'No frames provided'}), 400

    # Preprocess the frames received from the frame extraction
    preprocessed_frames = preprocess_frames(frames)  # Call the preprocessing function

    # After preprocessing, send the data to the main model API
    main_model_url = 'http://localhost:5000/main_model'  # Change this to your main model API URL
    response = requests.post(main_model_url, json={'preprocessed_frames': preprocessed_frames.tolist()})  # Convert to list for JSON

    if response.status_code != 200:
        return jsonify({'error': 'Main model processing failed', 'details': response.json()}), 500

    return jsonify({'message': 'Preprocessing completed successfully', 'model_result': response.json()}), 200
