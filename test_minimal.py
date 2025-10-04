import sys
import os

print("🚀 Démarrage de l'application Flask...")

# Ajouter le chemin courant pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Importations de base
    from flask import Flask, render_template, request, jsonify
    import pandas as pd
    import numpy as np
    import json
    
    print("✅ Importations de base réussies")
    
    # Importations optionnelles avec gestion d'erreur
    try:
        import xgboost as xgb
        print("✅ XGBoost importé")
    except ImportError as e:
        print(f"❌ XGBoost: {e}")
        xgb = None
        
    try:
        import plotly.graph_objects as go
        import plotly.utils
        print("✅ Plotly importé")
    except ImportError as e:
        print(f"❌ Plotly: {e}")
        go = None
        
    try:
        import joblib
        print("✅ Joblib importé")
    except ImportError as e:
        print(f"❌ Joblib: {e}")
        joblib = None
        
    print("✅ Toutes les importations terminées")
    
except Exception as e:
    print(f"❌ Erreur d'importation: {e}")
    input("Appuyez sur Entrée pour quitter...")
    sys.exit(1)

# Création de l'application Flask
app = Flask(__name__)

# Variables globales pour les modèles
modele_final = None
encoder = None
scaler = None
clf = None

def load_models():
    """Charge les modèles avec gestion d'erreur"""
    global modele_final, encoder, scaler, clf
    
    print("🔍 Chargement des modèles...")
    
    if joblib is None:
        print("❌ Joblib non disponible - mode démonstration activé")
        return False
        
    try:
        # Vérifier le dossier models
        if not os.path.exists('models'):
            print("❌ Dossier 'models' introuvable")
            return False
            
        # Liste des fichiers requis
        model_files = ['modele_final.pkl', 'encoder.pkl', 'scaler.pkl', 'clf.pkl']
        
        for filename in model_files:
            filepath = os.path.join('models', filename)
            if not os.path.exists(filepath):
                print(f"❌ Fichier manquant: {filepath}")
                return False
        
        # Charger les modèles
        print("📦 Chargement des modèles...")
        modele_final = joblib.load('models/modele_final.pkl')
        encoder = joblib.load('models/encoder.pkl')
        scaler = joblib.load('models/scaler.pkl') 
        clf = joblib.load('models/clf.pkl')
        
        print("✅ Tous les modèles chargés avec succès!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur de chargement des modèles: {e}")
        return False

def create_simple_plot(risk_level, risk_value, risk_color):
    """Crée un graphique simple"""
    if go is None:
        # Retourner des données simulées si Plotly n'est pas disponible
        return json.dumps({
            "data": [{
                "type": "indicator",
                "mode": "gauge+number", 
                "value": risk_value,
                "number": {"suffix": "% risque"},
                "title": {"text": f"Risque: {risk_level}"},
                "gauge": {
                    "axis": {"range": [0, 100]},
                    "bar": {"color": risk_color},
                    "steps": [
                        {"range": [0, 33], "color": "lightgreen"},
                        {"range": [33, 66], "color": "yellow"},
                        {"range": [66, 100], "color": "red"}
                    ]
                }
            }],
            "layout": {"height": 300}
        })
    
    # Version avec Plotly si disponible
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_value,
        number={"suffix": "% risque"},
        title={"text": f"Risque: {risk_level}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": risk_color},
            "steps": [
                {"range": [0, 33], "color": "lightgreen"},
                {"range": [33, 66], "color": "yellow"}, 
                {"range": [66, 100], "color": "red"}
            ]
        }
    ))
    fig.update_layout(height=300)
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# Charger les modèles au démarrage
models_loaded = load_models()

@app.route('/')
def index():
    """Page d'accueil principale"""
    print("📄 Requête reçue sur la page d'accueil")
    return render_template('index.html')

@app.route('/health')
def health():
    """Endpoint de santé"""
    return jsonify({
        'status': 'ok',
        'models_loaded': models_loaded,
        'message': 'Application Flask active'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction"""
    print("🎯 Requête de prédiction reçue")
    
    try:
        data = request.json
        print(f"📊 Données reçues: {data}")
        
        # Utiliser les modèles réels si disponibles
        if models_loaded and all([modele_final, encoder, scaler, clf]):
            print("🔮 Utilisation des modèles réels")
            
            # Préparation des données pour XGBoost
            client_data_xgb = pd.DataFrame({
                'age': [float(data['age'])],
                'bmi': [float(data['bmi'])], 
                'children': [int(data['children'])],
                'sex_female': [1 if data['sex'] == 'female' else 0],
                'sex_male': [1 if data['sex'] == 'male' else 0],
                'smoker_no': [1 if data['smoker'] == 'no' else 0],
                'smoker_yes': [1 if data['smoker'] == 'yes' else 0],
                'region_northeast': [1 if data['region'] == 'northeast' else 0],
                'region_northwest': [1 if data['region'] == 'northwest' else 0],
                'region_southeast': [1 if data['region'] == 'southeast' else 0],
                'region_southwest': [1 if data['region'] == 'southwest' else 0]
            })

            # Prédiction des frais
            frais_predits = modele_final.predict(xgb.DMatrix(client_data_xgb))[0]
            
        else:
            print("🎭 Utilisation des données simulées")
            # Données simulées pour la démonstration
            base_price = 10000
            age_factor = float(data['age']) * 100
            bmi_factor = float(data['bmi']) * 200
            smoker_factor = 5000 if data['smoker'] == 'yes' else 0
            children_factor = int(data['children']) * 500
            
            frais_predits = base_price + age_factor + bmi_factor + smoker_factor + children_factor
        
        frais_formatted = f"${frais_predits:,.2f}"
        print(f"💰 Frais prédits: {frais_formatted}")

        # Détermination du risque (simulée)
        risk_levels = [
            {"label": "Faible", "color": "#27ae60", "text": "Remboursement Élevé (80%)", "value": 20, "pack": "Pack Standard"},
            {"label": "Modéré", "color": "#f39c12", "text": "Remboursement Moyen (50%)", "value": 50, "pack": "Pack Premium"},
            {"label": "Élevé", "color": "#c0392b", "text": "Remboursement Faible (30%)", "value": 80, "pack": "Pack Premium Plus"}
        ]
        
        # Logique simple pour déterminer le risque
        if data['smoker'] == 'yes' or float(data['bmi']) > 30:
            risk = risk_levels[2]  # Élevé
        elif float(data['bmi']) > 25 or float(data['age']) > 50:
            risk = risk_levels[1]  # Modéré
        else:
            risk = risk_levels[0]  # Faible

        # Création du graphique
        graph_json = create_simple_plot(risk["label"], risk["value"], risk["color"])

        return jsonify({
            'success': True,
            'frais_predits': frais_formatted,
            'risk_label': risk["label"],
            'risk_color': risk["color"],
            'remboursement_text': risk["text"],
            'pack_recommande': risk["pack"],
            'graph_json': graph_json
        })
        
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/test')
def test_page():
    """Page de test simple"""
    return """
    <html>
        <head><title>Test Réussi</title></head>
        <body>
            <h1>✅ Application Flask Fonctionne!</h1>
            <p>Les routes principales:</p>
            <ul>
                <li><a href="/">Page d'accueil</a></li>
                <li><a href="/health">Health Check</a></li>
            </ul>
        </body>
    </html>
    """

if __name__ == '__main__':
    print("=" * 50)
    print("🎯 APPLICATION FLASK PRÊTE")
    print("=" * 50)
    print(f"📊 Modèles chargés: {models_loaded}")
    print("🌐 URL: http://localhost:5000")
    print("🔧 Debug: http://localhost:5000/test")
    print("❤️  Health: http://localhost:5000/health")
    print("⏹️  Ctrl+C pour arrêter")
    print("=" * 50)
    
    # Démarrer le serveur
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=False  # Éviter le double démarrage
    )
    