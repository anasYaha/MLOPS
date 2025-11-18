"""
Script principal pour exécuter le pipeline ML
"""
import argparse
import sys
from model_pipeline import (
    load_data,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model
)


def main():
    """Fonction principale avec arguments CLI"""
    parser = argparse.ArgumentParser(
        description='Pipeline ML pour classification d\'images avec SVM et HOG'
    )
    
    parser.add_argument(
        '--action',
        type=str,
        required=True,
        choices=['train', 'evaluate', 'predict', 'full'],
        help='Action à exécuter: train, evaluate, predict, ou full (pipeline complet)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='shipsnet.json',
        help='Chemin vers le fichier de données JSON (default: shipsnet.json)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='model.pkl',
        help='Chemin pour sauvegarder/charger le modèle (default: model.pkl)'
    )
    
    parser.add_argument(
        '--train-split',
        type=int,
        default=80,
        help='Pourcentage de données pour l\'entraînement (default: 80)'
    )
    
    parser.add_argument(
        '--ppc',
        type=int,
        default=16,
        help='Pixels per cell pour HOG features (default: 16)'
    )
    
    args = parser.parse_args()
    
    # Exécuter l'action demandée
    if args.action == 'full':
        print("=== Pipeline complet ===")
        run_full_pipeline(args)
    
    elif args.action == 'train':
        print("=== Entraînement du modèle ===")
        run_training(args)
    
    elif args.action == 'evaluate':
        print("=== Évaluation du modèle ===")
        run_evaluation(args)
    
    elif args.action == 'predict':
        print("=== Prédiction ===")
        run_prediction(args)


def run_full_pipeline(args):
    """Exécuter le pipeline complet"""
    print(f"\n1. Chargement des données depuis {args.data}...")
    data, labels = load_data(args.data)
    print(f"    {len(data)} images chargées")
    
    print(f"\n2. Prétraitement des données (PPC={args.ppc})...")
    data_frame = prepare_data(data, labels, ppc=args.ppc)
    print(f"    Features HOG extraits")
    
    print(f"\n3. Entraînement du modèle (split={args.train_split}%)...")
    clf, x_test, y_test = train_model(data_frame, train_percentage=args.train_split)
    print(f"    Modèle entraîné")
    
    print(f"\n4. Évaluation du modèle...")
    results = evaluate_model(clf, x_test, y_test)
    print(f"    Accuracy: {results['accuracy']:.4f}")
    
    print(f"\n5. Sauvegarde du modèle dans {args.model}...")
    save_model(clf, args.model)
    print(f"    Modèle sauvegardé")
    
    print("\n=== Pipeline terminé avec succès ===")


def run_training(args):
    """Entraîner et sauvegarder le modèle"""
    print(f"Chargement des données depuis {args.data}...")
    data, labels = load_data(args.data)
    
    print("Prétraitement des données...")
    data_frame = prepare_data(data, labels, ppc=args.ppc)
    
    print("Entraînement du modèle...")
    clf, x_test, y_test = train_model(data_frame, train_percentage=args.train_split)
    
    print(f"Sauvegarde du modèle dans {args.model}...")
    save_model(clf, args.model)
    
    print("Entraînement terminé!")


def run_evaluation(args):
    """Évaluer un modèle existant"""
    print(f"Chargement du modèle depuis {args.model}...")
    clf = load_model(args.model)
    
    print(f"Chargement des données de test depuis {args.data}...")
    data, labels = load_data(args.data)
    data_frame = prepare_data(data, labels, ppc=args.ppc)
    
    # Utiliser seulement les données de test
    partition = int(len(data_frame) * args.train_split / 100)
    x_test = data_frame[partition:, :-1]
    y_test = data_frame[partition:, -1:].ravel()
    
    print("Évaluation du modèle...")
    results = evaluate_model(clf, x_test, y_test)
    
    print("Évaluation terminée!")


def run_prediction(args):
    """Faire des prédictions avec un modèle existant"""
    print(f"Chargement du modèle depuis {args.model}...")
    clf = load_model(args.model)
    
    print(f"Chargement des données depuis {args.data}...")
    data, labels = load_data(args.data)
    data_frame = prepare_data(data, labels, ppc=args.ppc)
    
    # Faire des prédictions sur toutes les données
    x_data = data_frame[:, :-1]
    y_true = data_frame[:, -1:].ravel()
    
    print("Prédictions en cours...")
    y_pred = clf.predict(x_data)
    
    print(f"\nNombre total de prédictions: {len(y_pred)}")
    print(f"Prédictions positives (navires): {sum(y_pred == 1)}")
    print(f"Prédictions négatives: {sum(y_pred == 0)}")
    
    print("Prédictions terminées!")


if __name__ == "__main__":
    main()
