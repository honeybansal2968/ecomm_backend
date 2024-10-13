from flask import Flask,request,jsonify
from flask_cors import CORS
import base64
import io

import numpy as np
import Voice_to_Text,Data_Recommendation,Gemini_API
from PIL import Image
app = Flask(__name__)
import os
CORS(app, resources={r"/*": {"origins": "*"}})
Gemini_API.Start_a_Chat()
SAVE_DIR = 'saved_images'
os.makedirs(SAVE_DIR, exist_ok=True)
@app.route('/')
def home():
    return "Hello There "

@app.route('/voice_input')
def voice():
    text = request.args.get('text')
    # return Data_Recommendation.show_recommendation()
    return Data_Recommendation.show_recommendation(text)
@app.route('/text_input')
def textSearch():
    text = request.args.get('text')
    return Data_Recommendation.getRecommendedProducts(text)
@app.route('/transcript')
def transcriptAPI():
    path=request.args.get('Path')
    text_data=Voice_to_Text.speech_to_text(path)
    print(text_data)
    return text_data


@app.route('/image', methods=['POST'])
def image():
    try:
        data = request.json
        base64_string = data.get('imageData')
        if not base64_string:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode the base64 string
        image_data = base64.b64decode(base64_string)
        print("image_data", image_data[:20])  # Print a snippet for debugging

        # Load the image using PIL
        image = Image.open(io.BytesIO(image_data))
        
        # Convert image to RGB if necessary
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')
        
        # Create a unique filename
        filename = 'image.jpg'
        file_path = os.path.join(SAVE_DIR, filename)
        
        # Save the image data to a file
        with open(file_path, 'wb') as file:
            image.save(file, 'JPEG')
        
        print("Saved image to", file_path)
        
        # Call the Data_Recommendation function
        response_data = Data_Recommendation.show_image_recommendation(file_path)
        
        # Delete the saved image after processing
        if os.path.exists(file_path):
            os.remove(file_path)
            print("Deleted image from", file_path)
        
        # Return the response from Data_Recommendation
        return response_data
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/default')
def default_search():
    default_products=np.random.randint(1,100,50).tolist()
    return Data_Recommendation.get_data(default_products)
app.run(debug=True)