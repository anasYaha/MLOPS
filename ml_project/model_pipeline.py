"""
Pipeline modulaire pour l'entraînement d'un classificateur SVM avec HOG features
"""
import numpy as np
import json
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
import joblib


def load_data(filepath):
    """
    Charger les données depuis un fichier JSON contenant des images.
    
    Le fichier JSON shipsnet.json contient:
    - 'data': pixels des images (liste aplatie de valeurs RGB)
    - 'labels': étiquettes (0 = pas de navire, 1 = navire)
    - 'locations': coordonnées géographiques
    - 'scene_ids': identifiants des scènes
    
    Args:
        filepath (str): Chemin vers le fichier shipsnet.json
    
    Returns:
        tuple: (data, labels) 
            - data: Images RGB 80x80x3 (numpy array)
            - labels: Étiquettes correspondantes (numpy array)
    """
    print(f"Chargement du fichier JSON: {filepath}")
    
    with open(filepath, 'r') as f:
        dataset = json.load(f)
    
    print(f"  - Nombre d'images: {len(dataset['data'])}")
    print(f"  - Clés du JSON: {list(dataset.keys())}")
    
    # Les données sont stockées sous forme aplatie
    # Format: [R1, R2, ..., G1, G2, ..., B1, B2, ...]
    # Il faut les reformater en images 80x80x3 (RGB)
    data = np.array(dataset['data']).astype('uint8')
    img_length = 80
    
    # Reshape: (nombre_images, 3_canaux, 80, 80) -> (nombre_images, 80, 80, 3_canaux)
    data = data.reshape(-1, 3, img_length, img_length).transpose([0, 2, 3, 1])
    
    # Extraire les labels
    labels = np.array(dataset['labels']).reshape(len(dataset['labels']), 1)
    
    print(f"  ✓ Images reformatées: {data.shape}")
    print(f"  ✓ Labels extraits: {labels.shape}")
    
    return data, labels


def prepare_data(data, labels, ppc=16):
    """
    Prétraiter les données : conversion en grayscale et extraction HOG features.
    
    Args:
        data (np.array): Images RGB
        labels (np.array): Labels des images
        ppc (int): Pixels per cell pour HOG (default: 16)
    
    Returns:
        tuple: (hog_features, labels) - Features HOG et labels
    """
    # Conversion en grayscale
    data_gray = [color.rgb2gray(img) for img in data]
    
    # Extraction des features HOG
    hog_features = []
    for image in data_gray:
        fd, _ = hog(
            image,
            orientations=8,
            pixels_per_cell=(ppc, ppc),
            cells_per_block=(4, 4),
            block_norm='L2',
            visualize=True
        )
        hog_features.append(fd)
    
    hog_features = np.array(hog_features)
    
    # Combiner features et labels, puis mélanger
    data_frame = np.hstack((hog_features, labels))
    np.random.shuffle(data_frame)
    
    return data_frame


def train_model(data_frame, train_percentage=80):
    """
    Entraîner le modèle SVM sur les données.
    
    Args:
        data_frame (np.array): Données combinées (features + labels)
        train_percentage (int): Pourcentage de données pour l'entraînement
    
    Returns:
        tuple: (clf, x_test, y_test) - Modèle entraîné et données de test
    """
    # Partition des données
    partition = int(len(data_frame) * train_percentage / 100)
    
    x_train = data_frame[:partition, :-1]
    x_test = data_frame[partition:, :-1]
    y_train = data_frame[:partition, -1:].ravel()
    y_test = data_frame[partition:, -1:].ravel()
    
    # Entraînement du modèle SVM
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    
    print("Paramètres du modèle:", clf.get_params())
    
    return clf, x_test, y_test


def evaluate_model(clf, x_test, y_test):
    """
    Évaluer les performances du modèle.
    
    Args:
        clf: Modèle entraîné
        x_test (np.array): Données de test
        y_test (np.array): Labels de test
    
    Returns:
        dict: Métriques d'évaluation
    """
    y_pred = clf.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print("\nClassification Report:")
    print(report)
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'report': report
    }


def save_model(clf, filepath='model.pkl'):
    """
    Sauvegarder le modèle entraîné.
    
    Args:
        clf: Modèle à sauvegarder
        filepath (str): Chemin de sauvegarde (default: 'model.pkl')
    """
    joblib.dump(clf, filepath)
    print(f"Modèle sauvegardé dans {filepath}")


def load_model(filepath='model.pkl'):
    """
    Charger un modèle sauvegardé.
    
    Args:
        filepath (str): Chemin du modèle (default: 'model.pkl')
    
    Returns:
        Modèle chargé
    """
    clf = joblib.load(filepath)
    print(f"Modèle chargé depuis {filepath}")
    return clf
