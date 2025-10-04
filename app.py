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

modele_final, encoder, scaler, clf = load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if modele_final is None or encoder is None or scaler is None or clf is None:
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

        # Prédiction XGBoost
        client_data_xgb = pd.DataFrame({
            'age': [age], 'bmi': [bmi], 'children': [children],
            'sex_female': [1 if sex == 'female' else 0],
            'sex_male': [1 if sex == 'male' else 0],
            'smoker_no': [1 if smoker == 'no' else 0],
            'smoker_yes': [1 if smoker == 'yes' else 0],
            'region_northeast': [1 if region == 'northeast' else 0],
            'region_northwest': [1 if region == 'northwest' else 0],
            'region_southeast': [1 if region == 'southeast' else 0],
            'region_southwest': [1 if region == 'southwest' else 0]
        })

        frais_predits = modele_final.predict(xgb.DMatrix(client_data_xgb))[0]
        frais_formatted = f"${frais_predits:,.2f}"

        # Classification
        client_raw = pd.DataFrame([{
            'age': age, 'bmi': bmi, 'children': children,
            'sex': sex, 'smoker': smoker, 'region': region
        }])

        encoded = encoder.transform(client_raw[['sex', 'smoker', 'region']])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['sex', 'smoker', 'region']))
        scaled = scaler.transform(client_raw[['age', 'bmi', 'children']])
        scaled_df = pd.DataFrame(scaled, columns=['age', 'bmi', 'children'])
        client_for_tree = pd.concat([scaled_df, encoded_df], axis=1)

        remboursement_class = clf.predict(client_for_tree)[0]

        # Niveaux professionnels
        if remboursement_class == "R3":
            risk_data = {
                "label": "Faible Risque",
                "color": "#10B981",
                "taux_remboursement": 85,
                "pack": "Premium",
                "description": "Profil optimal - Couverture complète",
                "features": [
                    "Remboursement à 85%",
                    "Médecine courante et spécialisée",
                    "Hospitalisation complète",
                    "Pharmacie à 90%",
                    "Dentaire et optique inclus"
                ]
            }
        elif remboursement_class == "R2":
            risk_data = {
                "label": "Risque Modéré", 
                "color": "#F59E0B",
                "taux_remboursement": 65,
                "pack": "Confort",
                "description": "Profil standard - Bonne couverture",
                "features": [
                    "Remboursement à 65%",
                    "Médecine courante étendue",
                    "Hospitalisation partielle",
                    "Pharmacie à 70%",
                    "Dentaire de base"
                ]
            }
        else:
            risk_data = {
                "label": "Risque Élevé",
                "color": "#EF4444", 
                "taux_remboursement": 45,
                "pack": "Essentiel",
                "description": "Profil nécessitant une vigilance particulière",
                "features": [
                    "Remboursement à 45%",
                    "Médecine courante limitée",
                    "Hospitalisation d'urgence",
                    "Pharmacie à 50%",
                    "Soins essentiels"
                ]
            }

        # Graphique professionnel
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_data["taux_remboursement"],
            number={'suffix': "%", 'font': {'size': 36, 'color': "#1F2937", 'family': "Inter"}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Taux de Remboursement", 'font': {'size': 16, 'color': "#374151", 'family': "Inter"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#1F2937"},
                'bar': {'color': risk_data["color"], 'thickness': 0.6},
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "#E5E7EB",
                'steps': [
                    {'range': [0, 33], 'color': '#F3F4F6'},
                    {'range': [33, 66], 'color': '#F3F4F6'},
                    {'range': [66, 100], 'color': '#F3F4F6'}],
            }
        ))
        
        fig.update_layout(
            height=280,
            margin=dict(t=60, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'family': "Inter", 'color': "#1F2937"}
        )

        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return jsonify({
            'success': True,
            'frais_predits': frais_formatted,
            'risk_data': risk_data,
            'graph_json': graph_json
        })

    except Exception as e:
        logging.error(f"Erreur prédiction: {e}")
        return jsonify({'success': False, 'error': "Erreur lors de l'analyse. Veuillez réessayer."})

if __name__ == '__main__':
    print("Application démarrée")
    print("http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)