import os
import joblib

print("🔍 Vérification des modèles...")

# Vérifier le dossier models
if not os.path.exists('models'):
    print("❌ Le dossier 'models' n'existe pas")
    print("💡 Exécutez d'abord: python projetML.py")
    exit()

# Vérifier les fichiers
required_files = ['modele_final.pkl', 'encoder.pkl', 'scaler.pkl', 'clf.pkl']
all_exist = True

for file in required_files:
    file_path = f'models/{file}'
    if os.path.exists(file_path):
        print(f"✅ {file} - EXISTE")
        try:
            # Essayer de charger le fichier
            model = joblib.load(file_path)
            print(f"   📦 Chargé avec succès - Type: {type(model)}")
        except Exception as e:
            print(f"   ❌ Erreur de chargement: {e}")
            all_exist = False
    else:
        print(f"❌ {file} - MANQUANT")
        all_exist = False

if all_exist:
    print("\n🎯 Tous les modèles sont prêts!")
    print("➡️  Vous pouvez maintenant lancer: python app.py")
else:
    print("\n💡 Certains modèles manquent. Exécutez d'abord: python projetML.py")