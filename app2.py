from flask import Flask, render_template, request, jsonify
import pandas as pd
import xgboost as xgb
import plotly.graph_objects as go
import plotly.utils
import json
import joblib
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

def load_models():
    """Charge tous les modèles sauvegardés"""
    try:
        logging.info("Chargement des modèles...")
        
        if not os.path.exists('models'):
            logging.error("Dossier 'models' introuvable")
            return None, None, None, None
        
        required_files = ['modele_final.pkl', 'encoder.pkl', 'scaler.pkl', 'clf.pkl']
        for file in required_files:
            if not os.path.exists(f'models/{file}'):
                logging.error(f"Fichier manquant: models/{file}")
                return None, None, None, None
        
        modele_final = joblib.load('models/modele_final.pkl')
        encoder = joblib.load('models/encoder.pkl') 
        scaler = joblib.load('models/scaler.pkl')
        clf = joblib.load('models/clf.pkl')
        
        logging.info("Modèles chargés avec succès")
        return modele_final, encoder, scaler, clf
        
    except Exception as e:
        logging.error(f"Erreur: {e}")
        return None, None, None, None

# Charger les modèles
modele_final, encoder, scaler, clf = load_models()

# =============================================================================
# FONCTIONS DE PRÉDICTION AVEC VOTRE LOGIQUE
# =============================================================================

def prepare_client_data(age, bmi, children, sex, smoker, region):
    """Prépare les données du client dans le format attendu par le modèle"""
    
    # One-hot encoding manuel pour correspondre à X_transformed
    sex_female = 1 if sex == 'female' else 0
    sex_male = 1 if sex == 'male' else 0
    
    smoker_no = 1 if smoker == 'no' else 0
    smoker_yes = 1 if smoker == 'yes' else 0
    
    region_northeast = 1 if region == 'northeast' else 0
    region_northwest = 1 if region == 'northwest' else 0
    region_southeast = 1 if region == 'southeast' else 0
    region_southwest = 1 if region == 'southwest' else 0
    
    # Création du DataFrame avec les mêmes colonnes que pendant l'entraînement
    client_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi], 
        'children': [children],
        'sex_female': [sex_female],
        'sex_male': [sex_male],
        'smoker_no': [smoker_no],
        'smoker_yes': [smoker_yes],
        'region_northeast': [region_northeast],
        'region_northwest': [region_northwest],
        'region_southeast': [region_southeast],
        'region_southwest': [region_southwest]
    })
    
    return client_data

def definir_pack_auto(bmi, age, type_client):
    """
    Détermine automatiquement un pack attractif en fonction de :
    - bmi : BMI du client
    - age : âge du client  
    - type_client : 'FUMEURS' ou 'NON-FUMEURS'
    """
    # Définir le niveau du pack selon BMI
    if bmi > 35:
        niveau_pack = "VIP"
    elif bmi > 30:
        niveau_pack = "Premium Plus"
    elif bmi > 25:
        niveau_pack = "Premium"
    else:
        niveau_pack = "Standard"

    # Optionnel : suffixe selon âge
    if age > 55:
        age_suffix = "Senior"
    elif age > 40:
        age_suffix = "Adulte"
    else:
        age_suffix = "Jeune"

    # Construction du nom attractif
    pack = f"{niveau_pack} {age_suffix}"

    return pack

def assign_reimbursement_class_dynamic(age, bmi, smoker, children):
    """
    Détermine la classe de remboursement basée sur les caractéristiques du client
    R3: Faible risque → fort remboursement (85%)
    R2: Risque moyen → remboursement moyen (65%) 
    R1: Risque élevé → faible remboursement (45%)
    """
    score_risque = 0
    
    # Facteurs de risque (plus réalistes)
    if smoker == 'yes':
        score_risque += 40  # Fumeur = risque majeur
    
    if bmi > 35:
        score_risque += 30  # Obésité sévère
    elif bmi > 30:
        score_risque += 20  # Obésité
    elif bmi > 25:
        score_risque += 10  # Surpoids
    
    if age > 60:
        score_risque += 25  # Senior
    elif age > 45:
        score_risque += 15  # Adulte moyen
    elif age > 30:
        score_risque += 5   # Jeune adulte
    
    if children > 3:
        score_risque += 5   # Famille nombreuse
    
    # Détermination des classes basée sur le score de risque
    if score_risque >= 60:
        return "R1"  # Risque élevé → faible remboursement
    elif score_risque >= 30:
        return "R2"  # Risque moyen → remboursement moyen
    else:
        return "R3"  # Faible risque → fort remboursement

def get_remboursement_details(remboursement_class, type_client):
    """Retourne les détails du remboursement selon la classe"""
    if remboursement_class == "R3":
        return {
            'taux_remboursement': 85,
            'color': "#1DB954", 
            'label': "Faible Risque",
            'description': "Profil optimal - Couverture complète",
            'features': [
                "fort remboursement",
                "Médecine courante et spécialisée",
                "Hospitalisation complète", 
                "Pharmacie à 90%",
                "Dentaire et optique inclus"
            ]
        }
    elif remboursement_class == "R2":
        return {
            'taux_remboursement': 65,
            'color': "#FFA726",
            'label': "Risque Modéré",
            'description': "Profil standard - Bonne couverture",
            'features': [
                "remboursement moyen",
                "Médecine courante étendue",
                "Hospitalisation partielle",
                "Pharmacie à 70%", 
                "Dentaire de base"
            ]
        }
    else:  # R1
        return {
            'taux_remboursement': 45,
            'color': "#FF5252",
            'label': "Risque Élevé",
            'description': "Profil nécessitant une vigilance particulière",
            'features': [
                "faible remboursement",
                "Médecine courante limitée",
                "Hospitalisation d'urgence",
                "Pharmacie à 50%",
                "Soins essentiels"
            ]
        }

def predict_risk_and_pack(age, bmi, children, smoker):
    """Prédit le risque, le pack et le remboursement basé sur les données du client"""
    
    # Déterminer le type de client
    type_client = "FUMEURS" if smoker == 'yes' else "NON-FUMEURS"
    
    # Déterminer le pack selon votre logique
    pack = definir_pack_auto(bmi, age, type_client)
    
    # Déterminer la classe de remboursement selon la nouvelle logique
    remboursement_class = assign_reimbursement_class_dynamic(age, bmi, smoker, children)
    
    # Obtenir les détails du remboursement
    remboursement_details = get_remboursement_details(remboursement_class, type_client)
    
    # Description du profil
    if bmi > 35:
        bmi_label = "Obésité Sévère"
    elif bmi > 30:
        bmi_label = "Obésité" 
    elif bmi > 25:
        bmi_label = "Surpoids"
    else:
        bmi_label = "Poids Normal"
        
    if age > 55:
        age_label = "Senior"
    elif age > 40:
        age_label = "Adulte"
    else:
        age_label = "Jeune"
    
    profil = f"{bmi_label} + {age_label}"
    
    return {
        'pack': pack,
        'type_client': type_client,
        'remboursement_class': remboursement_class,
        'profil': profil,
        'taux_remboursement': remboursement_details['taux_remboursement'],
        'color': remboursement_details['color'],
        'label': remboursement_details['label'],
        'description': remboursement_details['description'],
        'features': remboursement_details['features'],
        'remboursement_text': f"Remboursement {remboursement_details['taux_remboursement']}%"
    }

def create_risk_gauge(taux_remboursement, color, label):
    """Crée un graphique jauge pour le type de remboursement"""
    
    # Déterminer le texte à afficher selon le taux
    if taux_remboursement >= 75:
        remboursement_text = "FORT"
        niveau_text = "Élevé"
    elif taux_remboursement >= 50:
        remboursement_text = "MOYEN" 
        niveau_text = "Modéré"
    else:
        remboursement_text = "FAIBLE"
        niveau_text = "Bas"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=taux_remboursement,
        number={
            'suffix': "", 
            'font': {'size': 36, 'color': "#FFFFFF", 'family': "Poppins"},
            'valueformat': '.0f'
        },
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"Type de Remboursement<br><span style='font-size:0.8em;color:{color}'>{remboursement_text}</span>", 
            'font': {'size': 16, 'color': "#FFFFFF", 'family': "Poppins"}
        },
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#191414", 'showticklabels': False},
            'bar': {'color': color, 'thickness': 0.6},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#E5E7EB",
            'steps': [
                {'range': [0, 33], 'color': '#F3F4F6'},
                {'range': [33, 66], 'color': '#F3F4F6'},
                {'range': [66, 100], 'color': '#F3F4F6'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': taux_remboursement
            }
        }
    ))
    
    # Ajouter des annotations pour les niveaux
    fig.add_annotation(
        x=0.15, y=0.1,
        text="FAIBLE",
        showarrow=False,
        font=dict(size=12, color="#FF5252" if taux_remboursement < 50 else "#9CA3AF"),
        xref="paper", yref="paper"
    )
    
    fig.add_annotation(
        x=0.5, y=0.1, 
        text="MOYEN",
        showarrow=False,
        font=dict(size=12, color="#FFA726" if 50 <= taux_remboursement < 75 else "#9CA3AF"),
        xref="paper", yref="paper"
    )
    
    fig.add_annotation(
        x=0.85, y=0.1,
        text="FORT", 
        showarrow=False,
        font=dict(size=12, color="#1DB954" if taux_remboursement >= 75 else "#9CA3AF"),
        xref="paper", yref="paper"
    )
    
    fig.update_layout(
        height=280,
        margin=dict(t=80, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': "Poppins", 'color': "#191414"}
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# =============================================================================
# ROUTES FLASK
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Route principale pour la prédiction des frais"""
    if modele_final is None:
        return jsonify({'success': False, 'error': "Système temporairement indisponible"})
    
    try:
        data = request.json
        
        age = int(data['age'])
        bmi = float(data['bmi'])
        children = int(data['children'])
        sex = data['sex']
        smoker = data['smoker']
        region = data['region']
        
        if age < 18 or age > 100:
            return jsonify({'success': False, 'error': "L'âge doit être entre 18 et 100 ans"})
        
        if children < 0 or children > 20:
            return jsonify({'success': False, 'error': "Nombre d'enfants invalide"})

        # ==================== PRÉDICTION DES FRAIS ====================
        client_data = prepare_client_data(age, bmi, children, sex, smoker, region)
        frais_predits = modele_final.predict(xgb.DMatrix(client_data))[0]
        frais_formatted = f"${frais_predits:,.2f}"

        # ==================== ÉVALUATION DU RISQUE ET PACK ====================
        risk_data = predict_risk_and_pack(age, bmi, children, smoker)
        
        # ==================== CRÉATION DU GRAPHIQUE ====================
        graph_json = create_risk_gauge(risk_data['taux_remboursement'], risk_data['color'], risk_data['label'])

        return jsonify({
            'success': True,
            'frais_predits': frais_formatted,
            'risk_data': risk_data,
            'graph_json': graph_json
        })

    except Exception as e:
        logging.error(f"Erreur prédiction: {e}")
        return jsonify({'success': False, 'error': f"Erreur lors de l'analyse: {str(e)}"})

@app.route('/pack', methods=['POST'])
def pack():
    """Route pour obtenir les détails du pack"""
    try:
        data = request.json
        
        age = int(data['age'])
        bmi = float(data['bmi'])
        children = int(data['children'])
        smoker = data['smoker']

        # ==================== ÉVALUATION DU RISQUE ET PACK ====================
        risk_data = predict_risk_and_pack(age, bmi, children, smoker)

        return jsonify({
            'success': True,
            'pack_data': {
                'pack': risk_data['pack'],
                'type_client': risk_data['type_client'],
                'profil': risk_data['profil'],
                'remboursement_class': risk_data['remboursement_class'],
                'taux_remboursement': risk_data['taux_remboursement'],
                'features': risk_data['features'],
                'description': risk_data['description'],
                'label': risk_data['label']
            }
        })

    except Exception as e:
        logging.error(f"Erreur pack: {e}")
        return jsonify({'success': False, 'error': f"Erreur lors de la détermination du pack: {str(e)}"})

if __name__ == '__main__':
    print("🚀 Application Flask démarrée")
    print("📍 http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)