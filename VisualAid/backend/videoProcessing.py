import cv2
import threading
from flask import Flask, request, jsonify
import requests
import tempfile
from queue import Queue
import time
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

from g4f.client import Client

client = Client()

def generatePara(s):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content":s+"These are the captions of the frames of a video. Give me a brief paragraph in simple language and nothing else in the response."}],
    )
    return response.choices[0].message.content

API_URL = os.getenv('API_URL')
HEADERS = os.getenv('API_TOKEN')

def query_model(frame):
    retry_attempts = 3
    retry_delay = 10
    for attempt in range(1, retry_attempts + 1):
        try:
            _, image_data = cv2.imencode('.jpg', frame)
            if isinstance(image_data, np.ndarray):
                image_data = image_data.tobytes()
            response = requests.post(API_URL, headers=HEADERS, data=image_data)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503 and "currently loading" in response.json().get("error", "").lower():
                estimated_time = response.json().get("estimated_time", 0)
                print(f"Model is still loading. Retry attempt {attempt}/{retry_attempts}. Waiting {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return {'error': f"Failed to query model: {response.status_code} {response.reason}"}
        except Exception as e:
            print(f"Exception occurred during model query: {str(e)}")
            return {'error': f"Exception occurred during model query: {str(e)}"}
    return {'error': f"Model did not become available after {retry_attempts} retries"}

def process_video(video_path, results_queue, completion_event):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        results_queue.put({'error': 'Failed to open video file.'})
        completion_event.set()
        return
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_seconds = 2
    interval_frames = int(frame_rate * interval_seconds)
    frame_number = 0
    while frame_number < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            try:
                result = query_model(frame)
                results_queue.put(result)
            except Exception as e:
                results_queue.put({'error': str(e)})
        frame_number += interval_frames
    cap.release()
    completion_event.set()

@app.route('/process_video', methods=['POST'])
def process_video_route():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided. Make sure to include a video file in the request.'}), 400
        video_file = request.files['video']
        temp_video_path = tempfile.mktemp(suffix='.mp4')
        video_file.save(temp_video_path)
        results_queue = Queue()
        completion_event = threading.Event()
        processing_thread = threading.Thread(target=process_video, args=(temp_video_path, results_queue, completion_event))
        processing_thread.start()
        completion_event.wait()
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        res = generatePara(str(results))
        return jsonify(res), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
