from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import gspread  # type: ignore # pour Google Sheets
import logging  # pour le logging
import os  # pour les variables d'environnement
import time  # Pour attendre en cas de problèmes Google Sheets

# Initialiser Flask
app = Flask(__name__)
CORS(app)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Charger le modèle de transformation de phrases
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Charger l'index FAISS
index = faiss.read_index("faiss_index.bin")

# Charger les questions/réponses
with open("qa_data.pkl", "rb") as f:
    data = pickle.load(f)
questions = data["questions"]
answers = data["answers"]

# Connexion à Google Sheet avec le compte de service
def connect_to_sheet():
    try:
        gc = gspread.service_account(filename="credentials.json")
        SHEET_NAME = os.environ.get("GOOGLE_SHEET_NAME", "chatbotAccess")  # Utiliser une variable d'environnement
        sh = gc.open(SHEET_NAME).sheet1  # Accéder à la feuille spécifiée
        logging.info(f"✅ Connecté à Google Sheet: {SHEET_NAME}")
        return sh
    except Exception as e:
        logging.error(f"❌ Erreur de connexion à Google Sheet: {e}")
        return None

sh = connect_to_sheet()  # Tenter la connexion une seule fois au démarrage

# Fonction pour trouver et remplir une cellule vide
def fill_first_empty_cell(worksheet, row_data):
    if not worksheet:
        logging.warning("⚠️ Impossible d'écrire dans Google Sheet (pas de connexion). Données perdues: {row_data}")
        return  # Sortir si pas de connexion

    try:
        cells = worksheet.get_all_values()  # Obtient toutes les valeurs de la feuille
        empty_row_found = -1

        for i, row in enumerate(cells):
            if i == 0:  # Ignorer la première ligne (en-têtes)
                continue
            if not row or all(not cell for cell in row):  # Ligne complètement vide
                empty_row_found = i + 1  # Index de la première ligne vide
                break
            elif not row[0] and not row[1]: #Les deux cellules sont vides
                empty_row_found = i + 1
                break
            elif not row[0]:  # Colonne A vide, Colonne B remplie
                empty_row_found = i + 1
                break
            elif not row[1]:  # Colonne B vide, Colonne A remplie
                empty_row_found = i + 1
                break

        if empty_row_found > 0:
            worksheet.update_cell(empty_row_found, 1, row_data[0])  # Remplir la colonne A (question)
            worksheet.update_cell(empty_row_found, 2, row_data[1])  # Remplir la colonne B (distance)
            logging.info(f"✅ Rempli la ligne {empty_row_found} avec: {row_data}")
        else:
            worksheet.append_row(row_data)
            logging.info(f"✅ Ajouté une nouvelle ligne à Google Sheet: {row_data}")

    except Exception as e:
        logging.error(f"❌ Erreur lors de l'écriture dans Google Sheet: {e}")
        # Tentative de reconnexion (une seule fois ici, vous pouvez ajouter une boucle de tentatives)
        global sh  # Pour modifier la variable globale
        sh = connect_to_sheet()
        if sh:
            try:
                sh.append_row(row_data)
                logging.info(f"✅ (Après reconnexion) Ajouté une nouvelle ligne à Google Sheet: {row_data}")
            except Exception as e2:
                logging.error(f"❌❌ Erreur après reconnexion: {e2} Données perdues: {row_data}")

# Fonction de recherche de la meilleure correspondance
def get_best_match(user_question):
    user_embedding = model.encode([user_question])
    D, I = index.search(np.array(user_embedding, dtype=np.float32), 1)
    distance = D[0][0]

    # Si la distance est élevée (> 0.5), enregistrer dans Google Sheet
    if distance > 0.5:
        if sh:  # Vérifier si la connexion à Google Sheets a réussi
            fill_first_empty_cell(sh, [user_question, str(distance)])  # Utiliser la nouvelle fonction
        else:
            logging.warning("⚠️ Impossible d'enregistrer la question dans Google Sheet (connexion échouée). Question: {user_question}")

        return "Je suis désolé 😞, je n'ai pas encore la réponse. Pourriez-vous la reformuler 🔄 ?"

    return answers[I[0][0]]

# Route principale /chat
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_question = data.get("question", "")
    if not user_question:
        return jsonify({"answer": "Veuillez poser une question."})

    response = get_best_match(user_question)
    return jsonify({"answer": response})

# Lancer l'application Flask
if __name__ == "__main__":
    logging.info("🚀 Démarrage de l'application Flask")
    app.run(debug=True)