from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from flask_cors import CORS
import requests
import base64
from io import BytesIO
from flask_bcrypt import Bcrypt
bcrypt = Bcrypt()
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config["MONGO_URI"] = os.getenv('DB_URL')
mongo = PyMongo(app)

API_URL = os.getenv('API_URL')
HEADERS = os.getenv('API_TOKEN')


def query_model(image_data):
    response = requests.post(API_URL, headers=HEADERS, data=image_data)
    return response.json()

@app.route('/')
def home():
    return "Welcome to the Flask MongoDB app!"

@app.route('/caption', methods=['POST'])
def get_image_caption():
    try:
        # Check if the request contains an image file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided. Make sure to include an image file in the request.'}), 400

        # Read the image file from the request
        image_file = request.files['image']
        image_file.seek(0)  # Ensure the file pointer is at the start

        # Check if the image file is empty
        image_content = image_file.read()
        if not image_content:
            return jsonify({'error': 'The provided image file is empty.'}), 400

        print("Image content length:", len(image_content))

        # Convert the image to base64
        image_base64 = base64.b64encode(image_content).decode('utf-8')
        print("Base64 encoded image:", image_base64[:100])  # Print first 100 characters for brevity

        # Query the model for image caption
        result = query_model(image_content)
        caption = result[0]["generated_text"]
        print("Generated caption:", caption)

        # Insert the data into MongoDB
        try:
            mongo.db.Assets.insert_one({"image_file": image_base64, "caption": caption})
            print("Inserted into database")
        except Exception as e:
            print(f"Error while uploading the conversation to the database: {e}")

        return jsonify(result[0]["generated_text"])

    except Exception as e:
        return jsonify({'error': str(e)}), 500

collection = mongo.db["Assets"]


@app.route('/conversations',methods = ['get'])
def send_conversations():
    print("Received fetch request")
    try:
        data = list(collection.find({}, {'_id': 0}))  # exclude _id field from the results
        print(jsonify(data))

        return jsonify(data)
    except Exception as e:
        print("Error while fetching data from database")





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


    