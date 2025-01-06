import logging
import os
from flask import Blueprint, jsonify, request
import requests


upload_video_api = Blueprint('upload_video_api', __name__)

# Configure upload folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@upload_video_api.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    video = request.files['video']
    title = video.filename
    email = request.form.get('email')

    if not video or video.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the video to the upload folder
    video_path = os.path.join("./uploads", video.filename)
    video.save(video_path)

    start_frame_extraction(video.filename, email)

    return jsonify({
        "message": "Video uploaded successfully",
        "video_path": video_path,
        "title": title,
        "email": email
    }), 200


def start_frame_extraction(title, email):
    frame_extraction_url= 'http://10.95.147.34:5000/api/frame_extraction_api/process_local_video'
    payload = {'title': title, 'email': email}
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(frame_extraction_url, json=payload, headers=headers, timeout=3000)
        response.raise_for_status()
        logging.debug("Received response from upload Video API.")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"frame Extraction API request failed: {e}")
        return {'error': f"frame Extraction API request failed: {e}"}