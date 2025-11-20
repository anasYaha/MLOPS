import argparse
import sys
import os
from model_pipeline import (
    prepare_data,
    split_data,
    train_base_model,
    hyperparameter_tuning,
    train_best_model,
    evaluate_model,
    create_confusion_matrix,
    plot_feature_importance,
    plot_sample_images,
    save_model,
    load_model,
    compare_models
)

def main():
    parser = argparse.ArgumentParser(description='Pipeline ML XGBoost pour classification de navires')
    parser.add_argument('--data_path', type=str, required=True, help='Chemin vers le fichier JSON des données')
    parser.add_argument('--test_size', type=float, default=0.2, help='Taille du jeu de test (défaut: 0.2)')
    parser.add_argument('--save_model', type=str, help='Chemin pour sauvegarder le modèle')
    parser.add_argument('--load_model', type=str, help='Chemin pour charger un modèle existant')
    parser.add_argument('--tuning', action='store_true', help='Effectuer la recherche d\'hyperparamètres')
    parser.add_argument('--visualize', action='store_true', help='Générer les visualisations')
    parser.add_argument('--cv_folds', type=int, default=5, help='Nombre de folds pour la validation croisée (défaut: 5)')
    
    args = parser.parse_args()

    # Vérifier que le fichier de données existe
    if not os.path.exists(args.data_path):
        print(f"Erreur: Le fichier {args.data_path} n'existe pas!")
        return

    if args.load_model:
        # Mode évaluation seulement
        print("=== MODE ÉVALUATION ===")
        model = load_model(args.load_model)
        features, labels, _, _, _ = prepare_data(args.data_path)
        X_train, X_test, y_train, y_test = split_data(features, labels, args.test_size)
        evaluate_model(model, X_test, y_test, "modèle chargé")
        
    else:
        # Mode entraînement complet
        print("=== MODE ENTRAÎNEMENT ===")
        
        # Préparation des données
        features, labels, data, data_gray, hog_images = prepare_data(args.data_path)
        X_train, X_test, y_train, y_test = split_data(features, labels, args.test_size)
        
        if args.visualize:
            print("\n=== GÉNÉRATION DES VISUALISATIONS ===")
            plot_sample_images(data, data_gray, hog_images, 'sample_images_xgboost.png')
        
        # Entraînement du modèle de base
        print("\n=== MODÈLE DE BASE ===")
        base_model, base_cv_scores = train_base_model(X_train, y_train, args.cv_folds)
        y_pred_base, accuracy_base = evaluate_model(base_model, X_test, y_test, "modèle de base")
        
        best_model = base_model
        
        if args.tuning:
            # Recherche d'hyperparamètres
            print("\n=== RECHERCHE D'HYPERPARAMÈTRES ===")
            grid_search = hyperparameter_tuning(X_train, y_train, args.cv_folds)
            best_model = train_best_model(grid_search, X_train, y_train)
            
            # Évaluation du meilleur modèle
            print("\n=== MEILLEUR MODÈLE ===")
            y_pred_best, accuracy_best = evaluate_model(best_model, X_test, y_test, "meilleur modèle")
            
            # Comparaison des modèles
            comparison = compare_models(base_model, best_model, X_test, y_test, X_train, y_train)
            
            # Visualisations du meilleur modèle
            if args.visualize:
                create_confusion_matrix(y_test, y_pred_best, 'confusion_matrix_xgboost.png')
                plot_feature_importance(best_model, 'feature_importance_xgboost.png')
        
        else:
            # Visualisations du modèle de base
            if args.visualize:
                create_confusion_matrix(y_test, y_pred_base, 'confusion_matrix_xgboost.png')
                plot_feature_importance(base_model, 'feature_importance_xgboost.png')
        
        # Sauvegarde du modèle
        if args.save_model:
            save_model(best_model, args.save_model)

if __name__ == "__main__":
    main()
