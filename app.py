from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
import pickle
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os

app = Flask(__name__)

# Diretório para salvar as imagens carregadas
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Carregar o modelo SVM e o label encoder
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Carregar o modelo VGG16 para extração de características
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img)
    return features.flatten()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Salvar a imagem carregada
        image_file = request.files['image']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)

        # Extrair características e classificar
        features = extract_features(image_path)
        features = np.expand_dims(features, axis=0)
        prediction = svm_model.predict(features)
        label = label_encoder.inverse_transform(prediction)[0]

        # Exibir a imagem e o resultado
        image_url = url_for('uploaded_file', filename=image_file.filename)

        return render_template('result.html', label=label, image_url=image_url)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
