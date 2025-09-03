import os
import joblib
import pandas as pd

# -------------------------
# Gestionnaire de modèle
# -------------------------
class ModelHandler:
    def __init__(self, model_path, scaler_path, feature_names):
        # Convertir le chemin pour supporter ~
        model_path = os.path.expanduser(model_path)
        scaler_path = os.path.expanduser(scaler_path)

        # Vérifier que les fichiers existent
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fichier introuvable : {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Fichier introuvable : {scaler_path}")

        # Charger le modèle et le scaler
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = feature_names

    def predict(self, X):
        # Transformer les données en DataFrame avec les bons noms de colonnes
        df = pd.DataFrame([X], columns=self.feature_names)
        X_scaled = self.scaler.transform(df)
        prediction = self.model.predict(X_scaled)
        return prediction[0], df

# -------------------------
# Noms des caractéristiques (features)
# -------------------------
FEATURE_NAMES = [
    'views', 'likes', 'dislikes', 'comment_count',
    'like_ratio', 'engagement_rate', 'title_length',
    'publish_hour', 'category_id'
]

# -------------------------
# Classes pour chaque modèle
# -------------------------
class RF(ModelHandler):
    def __init__(self):
        super().__init__("model/best_youtube_classifier_rfc.pkl",
                         "model/feature_scaler.pkl",
                         FEATURE_NAMES)

class SVM(ModelHandler):
    def __init__(self):
        super().__init__("model/best_youtube_classifier_svm.pkl",
                         "model/feature_scaler.pkl",
                         FEATURE_NAMES)

class XGB(ModelHandler):
    def __init__(self):
        super().__init__("model/best_youtube_classifier_xgb.pkl",
                         "model/feature_scaler.pkl",
                         FEATURE_NAMES)

class MLP(ModelHandler):
    def __init__(self):
        super().__init__("model/best_youtube_classifier_mlp.pkl",
                         "model/feature_scaler.pkl",
                         FEATURE_NAMES)

# -------------------------
# Interface terminal
# -------------------------
def choose_model():
    print("Choisissez le modèle pour la prédiction :")
    print("1 - Random Forest (RF)")
    print("2 - Support Vector Machine (SVM)")
    print("3 - XGBoost (XGB)")
    print("4 - Multi-Layer Perceptron (MLP)")
    choice = input("Entrez le numéro du modèle (1-4) : ").strip()
    if choice == '1':
        return RF()
    elif choice == '2':
        return SVM()
    elif choice == '3':
        return XGB()
    elif choice == '4':
        return MLP()
    else:
        print("Choix invalide. Veuillez réessayer.")
        return choose_model()

def get_user_features(feature_names):
    print("\nEntrez les valeurs des caractéristiques (features) :")
    user_values = []

    for f in feature_names:
        while True:
            # Ajout d'une aide spéciale pour category_id
            if f == 'category_id':
                print("\nNOTE : category_id correspond à la catégorie YouTube de la vidéo.")
                print("Exemples : 24 - Entertainment, 10 - Music, 26 - Howto & Style, 23 - Comedy, 22 - People & Blogs, 25 - News & Politics, 28 - Science & Technology, 1 - Film & Animation, 17 - Sports, 27 - Education, 15 - Pets & Animals, 20 - Gaming, 19 - Travel & Events, 2 - Autos & Vehicles, 29 - Nonprofits & Activism, 43 - Shows")
            val = input(f"{f}: ").strip()
            try:
                val = float(val)
                user_values.append(val)
                break
            except ValueError:
                print("Erreur : veuillez entrer un nombre.")
    return user_values

# -------------------------
# Programme principal
# -------------------------
if __name__ == "__main__":
    # Choisir le modèle
    model = choose_model()
    
    # Récupérer les valeurs des caractéristiques auprès de l'utilisateur
    user_input = get_user_features(model.feature_names)
    
    # Faire la prédiction
    pred, df = model.predict(user_input)

    # Afficher les données saisies
    print("\nCaractéristiques saisies :")
    print(df.to_string(index=False))
    
    # Afficher le résultat
    CLASS_LABELS = {
    0: "Faible popularité",
    1: "Popularité moyenne",
    2: "Haute popularité"
    }
    label = CLASS_LABELS.get(pred, "Classe inconnue")
    print(f"\nClasse prédite : {pred} ({label})")