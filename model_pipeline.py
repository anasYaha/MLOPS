import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Pour éviter les problèmes d'affichage en headless
import matplotlib.pyplot as plt
from skimage import color
from skimage.feature import hog
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
import joblib
import seaborn as sns
import os

def prepare_data(json_path):
    """Charge et prétraite les données avec extraction des features HOG"""
    print("Chargement des données...")
    with open(json_path) as f:
        dataset = json.load(f)
    
    data = np.array(dataset['data']).astype('uint8')
    img_length = 80
    data = data.reshape(-1, 3, img_length, img_length).transpose([0, 2, 3, 1])
    
    # Conversion en niveaux de gris
    print("Conversion en niveaux de gris...")
    data_gray = [color.rgb2gray(img) for img in data]
    
    # Extraction des features HOG
    print("Extraction des features HOG...")
    ppc = 16
    hog_features = []
    hog_images = []
    
    for image in data_gray:
        fd, hog_image = hog(image, 
                           orientations=8, 
                           pixels_per_cell=(ppc, ppc),
                           cells_per_block=(4, 4), 
                           block_norm='L2', 
                           visualize=True)
        hog_features.append(fd)
        hog_images.append(hog_image)
    
    labels = np.array(dataset['labels'])
    print(f"Données préparées: {len(hog_features)} échantillons")
    print(f"Longueur du vecteur de features: {len(hog_features[0])}")
    
    return np.array(hog_features), labels, data, data_gray, hog_images

def split_data(features, labels, test_size=0.2, random_state=42):
    """Divise les données en ensembles d'entraînement et de test"""
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels
    )
    
    print(f"Training set: {X_train.shape[0]} échantillons")
    print(f"Testing set: {X_test.shape[0]} échantillons")
    print(f"Distribution des classes: {np.unique(y_train, return_counts=True)}")
    
    return X_train, X_test, y_train, y_test

def train_base_model(X_train, y_train, cv_folds=5):
    """Entraîne un modèle XGBoost de base avec validation croisée"""
    print("Entraînement du modèle XGBoost de base...")
    
    base_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Validation croisée
    cv_scores = cross_val_score(base_model, X_train, y_train, cv=cv_folds, scoring='accuracy')
    
    print("Scores de validation croisée:", cv_scores)
    print(f"Accuracy moyenne: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Entraînement final
    base_model.fit(X_train, y_train)
    
    return base_model, cv_scores

def hyperparameter_tuning(X_train, y_train, cv_folds=5):
    """Recherche les meilleurs hyperparamètres avec GridSearchCV"""
    print("Recherche des meilleurs hyperparamètres...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv_folds,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("GridSearchCV terminé!")
    print(f"Meilleurs paramètres: {grid_search.best_params_}")
    print(f"Meilleur score: {grid_search.best_score_:.4f}")
    
    return grid_search

def train_best_model(grid_search, X_train, y_train):
    """Entraîne le meilleur modèle avec les paramètres optimisés"""
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    print("Meilleur modèle entraîné!")
    return best_model

def evaluate_model(model, X_test, y_test, model_name="Modèle"):
    """Évalue les performances du modèle"""
    print(f"Évaluation du {model_name}...")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred, target_names=['No Ship', 'Ship']))
    
    return y_pred, accuracy

def create_confusion_matrix(y_test, y_pred, save_path=None):
    """Crée et affiche la matrice de confusion"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Ship', 'Ship'], 
                yticklabels=['No Ship', 'Ship'])
    plt.title('Matrice de Confusion - XGBoost')
    plt.ylabel('Vraie étiquette')
    plt.xlabel('Étiquette prédite')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Matrice de confusion sauvegardée: {save_path}")
    
    plt.show()
    return cm

def plot_feature_importance(model, save_path=None):
    """Affiche l'importance des features"""
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, max_num_features=20, importance_type='weight')
    plt.title('Importance des Features - XGBoost')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Importance des features sauvegardée: {save_path}")
    
    plt.show()

def plot_sample_images(data, data_gray, hog_images, save_path=None):
    """Affiche des exemples d'images"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(data[51])
    plt.title('Image Originale')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(data_gray[51], cmap='gray')
    plt.title('Image Niveaux de Gris')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(hog_images[51], cmap='gray')
    plt.title('Features HOG')
    plt.axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Exemples d'images sauvegardés: {save_path}")
    
    plt.show()

def save_model(model, path):
    """Sauvegarde le modèle entraîné"""
    joblib.dump(model, path)
    print(f"Modèle sauvegardé: {path}")

def load_model(path):
    """Charge un modèle sauvegardé"""
    print(f"Chargement du modèle: {path}")
    return joblib.load(path)

def compare_models(base_model, best_model, X_test, y_test, X_train, y_train):
    """Compare les performances des modèles de base et optimisé"""
    print("=== COMPARAISON DES MODÈLES ===")
    
    # Prédictions
    y_pred_base = base_model.predict(X_test)
    y_pred_best = best_model.predict(X_test)
    
    # Accuracy
    accuracy_base = accuracy_score(y_test, y_pred_base)
    accuracy_best = accuracy_score(y_test, y_pred_best)
    
    print(f"Accuracy modèle de base: {accuracy_base:.4f}")
    print(f"Accuracy meilleur modèle: {accuracy_best:.4f}")
    print(f"Amélioration: {accuracy_best - accuracy_base:.4f}")
    
    # Validation croisée
    cv_scores_base = cross_val_score(base_model, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores_best = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"\nValidation croisée - Modèle de base: {cv_scores_base.mean():.4f} (+/- {cv_scores_base.std() * 2:.4f})")
    print(f"Validation croisée - Meilleur modèle: {cv_scores_best.mean():.4f} (+/- {cv_scores_best.std() * 2:.4f})")
    
    return {
        'base_accuracy': accuracy_base,
        'best_accuracy': accuracy_best,
        'improvement': accuracy_best - accuracy_base
    }
