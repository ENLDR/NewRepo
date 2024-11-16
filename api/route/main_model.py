# main_model.py
from flask import Blueprint, jsonify, request, current_app
import os
import uuid
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import base64  # For decoding base64 images
import mediapipe as mp
from tensorflow.keras.preprocessing.image import img_to_array
import time
from db_config import get_db, close_db



main_model_api = Blueprint('main_model_api', __name__)

# Global variable for the model and other constants
model = None
IMG_SIZE = (384, 512)  # Image size expected by the model
class_names = [
    'hachijiDachi_jodanYoko', 'sanchinDachi_ageUke', 'sanchinDachi_jodanTsuki', 
    'sanchinDachi_sotoUke', 'shikoDachi_gedanBarai', 'sotoUke_maeGeri', 
    'zenkutsuDachi_awaseTsuki', 'zenkutsuDachi_chudanTsuki', 'zenkutsuDachi_empiUke'
]
# Import the class_angles dictionary
class_angles = {
        'hachijiDachi_jodanYoko': [
        [178.6704642, 121.4324436, 178.6704642, 121.4324436, 176.2882244, 176.2882244, 179.0472569, 155.9306886, 174.7958128, 0],
        [74.73670094, 179.6333059, 74.73670094, 179.6333059, 179.2610734, 179.2610734, 177.7071013, 157.4409467, 146.1440207, 0]
    ],
    'sanchinDachi_ageUke': [
        [117.6311735, 177.122714, 117.6311735, 177.122714, 172.4470531, 172.4470531, 179.84531, 178.0124464, 154.3915227, 0],
        [88.35833, 130.8359801, 88.35833, 130.8359801, 178.8655222, 178.8655222, 179.2608471, 159.6653805, 172.8783815, 0]
    ],
    'sanchinDachi_jodanTsuki': [
        [79.67428505, 117.6143462, 79.67428505, 117.6143462, 175.4425771, 175.4425771, 179.9083308, 179.6819875, 162.7090183, 0],
        [132.2753167, 61.93098373, 132.2753167, 61.93098373, 177.3881788, 177.3881788, 177.6381035, 156.7447632, 179.4797426, 0]
    ],
    'sanchinDachi_sotoUke': [
        [11.47716376, 114.5169405, 11.47716376, 114.5169405, 176.3834111, 176.3834111, 175.592297, 179.2147185, 163.3744971, 0],
        [87.20938557, 16.70051146, 87.20938557, 16.70051146, 178.365065, 178.365065, 174.4879716, 162.4809264, 176.8140283, 0]
    ],
    'shikoDachi_gedanBarai': [
        [176.4795284, 153.3314441, 176.4795284, 153.3314441, 107.3410318, 107.3410318, 97.62956052, 153.8734055, 156.7703905, 0],
        [147.1951634, 175.074177, 147.1951634, 175.074177, 103.3349196, 103.3349196, 128.5222031, 157.1361985, 145.8509965, 0]
    ],
    'sotoUke_maeGeri': [
        [141.6715664, 12.61353582, 141.6715664, 12.61353582, 177.0001987, 177.0001987, 175.727286, 164.0403825, 160.4981336, 0],
        [15.39189415, 84.51329129, 15.39189415, 84.51329129, 174.1109382, 174.1109382, 158.9654381, 178.2361718, 159.3976762, 0]
    ],
    'zenkutsuDachi_awaseTsuki': [
        [115.8580351, 130.7511613, 115.8580351, 130.7511613, 171.9446945, 171.9446945, 172.6747746, 144.105041, 178.9400008, 0]
    ],
    'zenkutsuDachi_chudanTsuki': [
        [115.8580351, 130.7511613, 115.8580351, 130.7511613, 171.9446945, 171.9446945, 172.6747746, 144.105041, 178.9400008, 0],
        [156.0357435, 61.09435279, 156.0357435, 61.09435279, 172.3895637, 172.3895637, 162.5085944, 157.2766185, 159.0072517, 0]
    ],
    'zenkutsuDachi_empiUke': [
        [5.537744135, 138.5696075, 5.537744135, 138.5696075, 176.3914556, 176.3914556, 173.9572285, 166.2690671, 157.8046774, 0],
        [156.0357435, 61.09435279, 156.0357435, 61.09435279, 172.3895637, 172.3895637, 162.5085944, 157.2766185, 159.0072517, 0]
    ]
}


@main_model_api.before_app_first_request
def load_model():
    global model
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'cnn_model_augmented_dropout2.keras')  # Adjust model path as needed
    model = load_model(model_path)

@main_model_api.before_request
def setup_image_folder():
    global IMAGE_FOLDER, TEMP_IMAGE_FOLDER
    IMAGE_FOLDER = os.path.join(current_app.root_path, 'static', 'images')
    TEMP_IMAGE_FOLDER = os.path.join(current_app.root_path, 'static', 'temp_images')
    
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        
    if not os.path.exists(TEMP_IMAGE_FOLDER):
        os.makedirs(TEMP_IMAGE_FOLDER)


# Preprocess image function
def preprocess_image(img):
    img = cv2.resize(img, IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Extract keypoints function using Mediapipe
def extract_keypoints(img):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32)
    return keypoints

# Angle extraction function
def extract_angles(keypoints):
    if keypoints is None:
        return None

    LEFT_SHOULDER = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
    RIGHT_SHOULDER = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
    LEFT_ELBOW = mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value
    RIGHT_ELBOW = mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value
    LEFT_WRIST = mp.solutions.pose.PoseLandmark.LEFT_WRIST.value
    RIGHT_WRIST = mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value
    LEFT_HIP = mp.solutions.pose.PoseLandmark.LEFT_HIP.value
    RIGHT_HIP = mp.solutions.pose.PoseLandmark.RIGHT_HIP.value
    LEFT_KNEE = mp.solutions.pose.PoseLandmark.LEFT_KNEE.value
    RIGHT_KNEE = mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value
    LEFT_ANKLE = mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value
    RIGHT_ANKLE = mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value

    # Calculate angles between specific points
    left_shoulder_angle = calculate_angle(keypoints[LEFT_ELBOW], keypoints[LEFT_SHOULDER], keypoints[LEFT_HIP])
    right_shoulder_angle = calculate_angle(keypoints[RIGHT_HIP], keypoints[RIGHT_SHOULDER], keypoints[RIGHT_ELBOW])
    left_elbow_angle = calculate_angle(keypoints[LEFT_SHOULDER], keypoints[LEFT_ELBOW], keypoints[LEFT_WRIST])
    right_elbow_angle = calculate_angle(keypoints[RIGHT_WRIST], keypoints[RIGHT_ELBOW], keypoints[RIGHT_SHOULDER])
    left_waist_angle = calculate_angle(keypoints[LEFT_KNEE], keypoints[LEFT_HIP], keypoints[LEFT_SHOULDER])
    right_waist_angle = calculate_angle(keypoints[RIGHT_SHOULDER], keypoints[RIGHT_HIP], keypoints[RIGHT_KNEE])
    left_knee_angle = calculate_angle(keypoints[LEFT_HIP], keypoints[LEFT_KNEE], keypoints[LEFT_ANKLE])
    right_knee_angle = calculate_angle(keypoints[RIGHT_ANKLE], keypoints[RIGHT_KNEE], keypoints[RIGHT_HIP])
    left_ankle_angle = calculate_angle(keypoints[LEFT_KNEE], keypoints[LEFT_ANKLE], keypoints[LEFT_HIP])
    right_ankle_angle = calculate_angle(keypoints[RIGHT_HIP], keypoints[RIGHT_ANKLE], keypoints[RIGHT_KNEE])

    # Calculate angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


    # Return calculated angles as a list
    angles = [
        left_shoulder_angle, right_shoulder_angle,
        left_elbow_angle, right_elbow_angle,
        left_waist_angle, right_waist_angle,
        left_knee_angle, right_knee_angle,
        left_ankle_angle, right_ankle_angle
    ]
    
    return angles

# Compare angles function
def compare_angles(extracted_angles, predefined_angles):
    similarities = []
    for angle_set in predefined_angles:
        diff = np.abs(np.array(extracted_angles) - np.array(angle_set))
        similarity = np.mean(1 - (diff / 180.0))
        similarities.append(similarity * 100)
    return max(similarities)

@main_model_api.route('/main_model', methods=['POST'])
def main_model():
    try:
        data = request.json
        preprocessed_frames = data.get('preprocessed_frames')
        video_name = data.get('video_name')  # Assuming video name is sent in the request
        player_email = data.get('player_email')  # Assuming player's email is sent in the request

        if not preprocessed_frames or not video_name or not player_email:
            return jsonify({'error': 'Missing data'}), 400

        # Decode the base64-encoded frames
        decoded_images = []
        temp_image_paths = []  # Store paths to temporary images
        
        for i, frame in enumerate(preprocessed_frames):
            img_data = base64.b64decode(frame)
            np_img = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            # Save the image temporarily
            temp_image_path = os.path.join(TEMP_IMAGE_FOLDER, f"temp_image_{uuid.uuid4()}.jpg")
            cv2.imwrite(temp_image_path, img)
            temp_image_paths.append(temp_image_path)
            
            decoded_images.append(img)

        # Process images
        single_pose_results = []
        r_id = None
        for img, temp_path in zip(decoded_images, temp_image_paths):
            img_array = preprocess_image(img)
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_class_index]
            
            keypoints = extract_keypoints(img)
            extracted_angles = extract_angles(keypoints)
            class_name = class_names[predicted_class_index]
            predefined_angles = class_angles.get(class_name, [])
            
            if extracted_angles and predefined_angles:
                similarity = compare_angles(extracted_angles, predefined_angles)
            else:
                similarity = None

            if similarity is not None:
                if r_id is None:
                    r_id = save_result(video_name, player_email)
                
                save_single_pose(class_name, similarity, r_id)

                single_pose_results.append(similarity)

            # Delete the temporary image after processing
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Calculate the average of the single pose results (Final_result)
        if single_pose_results:
            final_result = np.mean(single_pose_results)
        else:
            final_result = 0

        # Update the result in the Result table
        update_result(r_id, final_result)

        return jsonify({'message': 'Processing complete', 'final_result': final_result}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
   # Save result to the database
def save_result(video_name, player_email):
    try:
        db = get_db()  # Get the database connection
        cursor = db.cursor()
        cursor.execute(''' 
            INSERT INTO Result (Video_name, Date, Final_result, Rank_P, P_email)
            VALUES (?, ?, ?, ?, ?)''', 
            (video_name, time.strftime('%Y-%m-%d'), 0, 'Not Ranked', player_email))
        db.commit()
        result_id = cursor.lastrowid
        cursor.close()
        return result_id
    except Exception as e:
        db.rollback()
        raise e



def save_single_pose(pose_name, pose_result, r_id):
    # Save individual pose result into the Single_pose table 
        try:
            db = get_db()
            cursor = db.cursor()
            # You might need to find the pose ID from Correct_pose table based on the pose name
            cursor.execute('SELECT Pose_id FROM Correct_pose WHERE Correct_Pose_name = %s', (pose_name,))
            c_id = cursor.fetchone()
        
            if c_id is None:
                # Handle case where the pose doesn't exist in Correct_pose (could add new pose if needed)
                raise ValueError(f"Pose name {pose_name} not found in Correct_pose table")

            cursor.execute('''
                INSERT INTO Single_pose (Pose_name, Single_pose_result, R_id, C_id)
                VALUES (%s, %s, %s, %s)
            ''', (pose_name, pose_result, r_id, c_id[0]))  # c_id[0] is the Pose_id from Correct_pose
            db.commit()
            cursor.close()
        except Exception as e:
            db.rollback()
            raise e


def update_result(r_id, final_result):
    # Update the Result table with the final result 
        try:
            db = get_db()
            cursor = db.cursor()
            cursor.execute('''
                UPDATE Result
                SET Final_result = %s
                WHERE Result_id = %s
            ''', (final_result, r_id))
            db.commit()
            cursor.close()
        except Exception as e:
            db.rollback()
            raise e


'''@main_model_api.route('/main_model', methods=['POST'])
def main_model():
    try:
        data = request.json
        preprocessed_frames = data.get('preprocessed_frames')

        if not preprocessed_frames:
            return jsonify({'error': 'No preprocessed frames provided'}), 400

        # Decode the base64-encoded frames
        decoded_images = []
        for frame in preprocessed_frames:
            img_data = base64.b64decode(frame)  # Decode the base64 string to binary
            np_img = np.frombuffer(img_data, dtype=np.uint8)  # Convert binary data into a NumPy array
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)  # Decode into an image using OpenCV
            decoded_images.append(img)

        # Process each decoded image
        results = []
        for img in decoded_images:
            # Preprocess the image for model input
            img_array = preprocess_image(img)

            # Predict class and get confidence
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_class_index]

            # Get keypoints and angles
            keypoints = extract_keypoints(img)
            extracted_angles = extract_angles(keypoints)
    
            # Calculate similarity if angles are extracted and class has predefined angles
            class_name = class_names[predicted_class_index]
            predefined_angles = class_angles.get(class_name, [])
            if extracted_angles and predefined_angles:
                similarity = compare_angles(extracted_angles, predefined_angles)
            else:
                similarity = None

            # Save the result into the database
            if similarity is not None:
             # Assuming r_id and c_id are available, possibly from the request or other sources
                r_id = 1  # Replace with actual value
                c_id = 1  # Replace with actual value

                # Call the save_single_pose function to save to the database
                save_single_pose(class_name, similarity, r_id, c_id)

            results.append({
                'predicted_class': class_name,
                'confidence': confidence,
                'similarity': similarity
            })
        for img in decoded_images:
            # Preprocess the image for model input
            img_array = preprocess_image(img)

            # Predict class and get confidence
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_class_index]

            # Get keypoints and angles
            keypoints = extract_keypoints(img)
            extracted_angles = extract_angles(keypoints)
            
            # Calculate similarity if angles are extracted and class has predefined angles
            class_name = class_names[predicted_class_index]
            predefined_angles = class_angles.get(class_name, [])
            if extracted_angles and predefined_angles:
                similarity = compare_angles(extracted_angles, predefined_angles)
            else:
                similarity = None

            results.append({
                'predicted_class': class_name,
                'confidence': confidence,
                'similarity': similarity
            })


        return jsonify({'results': results}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500'''


@main_model_api.route('/delete_image', methods=['POST'])
def delete_image():
    single_pose_id = request.json.get('single_pose_id')
    if not single_pose_id:
        return jsonify({'error': 'No single_pose_id provided'}), 400

    try:
        filepath = get_image_filepath(single_pose_id)
        if not filepath:
            return jsonify({'error': 'Image not found'}), 404

        if os.path.exists(filepath):
            os.remove(filepath)
            delete_single_pose(single_pose_id)
            return jsonify({'message': 'Image deleted successfully'}), 200
        else:
            return jsonify({'error': 'Image file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_image_filepath(single_pose_id):
    filename = f'{single_pose_id}.jpg'
    filepath = os.path.join(IMAGE_FOLDER, filename)
    return filepath

def delete_single_pose(single_pose_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute('DELETE FROM Single_pose WHERE Single_pose_id = %s', (single_pose_id,))
    db.commit()
    cursor.close()
