import numpy as np
import os
import cv2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Definir a função extract_features
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

# Caminhos e categorias
data_dir = os.path.expanduser("~/Desktop/faceid")
categories = ['cotista', 'nao_cotista']

# Preparar dados
features = []
labels = []

for category in categories:
    category_path = os.path.join(data_dir, category)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        feature = extract_features(img_path)
        features.append(feature)
        labels.append(category)

# Convertendo para numpy arrays
features = np.array(features)
labels = np.array(labels)

# Codificando as labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Dividindo os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Treinando o classificador SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

# Avaliando o modelo
y_pred = svm.predict(X_test)

# Acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.2f}")

# Relatório de Classificação
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Matriz de Confusão
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Salvando o modelo e o codificador de labels
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)