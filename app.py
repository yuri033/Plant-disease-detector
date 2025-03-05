import os
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('C://Users//Mayuri//Downloads//plant-disease-detect//model//PlantVillage_model.h5')

# Extract class labels correctly
DATASET_PATH = "C://Users//Mayuri//Downloads//plant-disease-detect//PlantVillage"

if os.path.exists(DATASET_PATH):
    class_labels = sorted(os.listdir(DATASET_PATH))
    class_labels = [folder for folder in class_labels if os.path.isdir(os.path.join(DATASET_PATH, folder))]
else:
    class_labels = ["Unknown Class"]

print(f"✅ Loaded Class Labels: {class_labels}")

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Image preprocessing function
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((126, 126))
    img = np.array(img) / 255.0
    img = img.reshape((1, 126, 126, 3))
    return img

# Route to render homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess and predict
    img = preprocess_image(file_path)
    predictions = model.predict(img)
    
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3_classes = [(class_labels[i], float(predictions[0][i]) * 100) for i in top_3_indices]

    predicted_class, confidence = top_3_classes[0]

    return render_template('result.html', filename=file.filename, prediction=predicted_class, confidence=confidence, top_3=top_3_classes)

# Route to serve uploaded images
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
