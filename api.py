# Phase 5b — API Flask avec ngrok
from flask import Flask, request, jsonify
from pyngrok import ngrok
import threading
import joblib
import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target

def split_train_val_test(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Découpe X, y en trois jeux : train, validation, test.
    Renvoie 6 objets : X_train, X_val, X_test, y_train, y_val, y_test.
    Les proportions sont respectées avec stratify=y.
    """
    if val_size <= 0 or val_size >= 1:
        raise ValueError(f"val_size doit être entre 0 et 1 (reçu : {val_size})")
    if test_size <= 0 or test_size >= 1:
        raise ValueError(f"test_size doit être entre 0 et 1 (reçu : {test_size})")

    # Premier split : isoler le test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Recalculer la proportion val sur le reste
    val_size_adjusted = val_size / (1 - test_size)

    # Deuxième split : isoler la validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)

# Charger le modèle
payload_api = joblib.load("modele.joblib")
modele_api  = payload_api["modele"]
scaler_api  = payload_api["scaler"]
target_names = payload_api["target_names"]
n_features   = len(payload_api["feature_names"])

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data_req = request.get_json(silent=True)

    # Validation
    if data_req is None:
        return jsonify({"error": "Corps JSON invalide ou manquant"}), 400
    if "features" not in data_req:
        return jsonify({"error": "Clé 'features' manquante"}), 400

    features = data_req["features"]

    if not isinstance(features, list) or len(features) == 0:
        return jsonify({"error": "features doit être une liste non vide"}), 400
    if len(features) != n_features:
        return jsonify({"error": f"Attendu {n_features} features, reçu {len(features)}"}), 400

    try:
        arr = np.array(features, dtype=float).reshape(1, -1)
    except (ValueError, TypeError):
        return jsonify({"error": "features doit contenir uniquement des nombres"}), 400

    arr_scaled = scaler_api.transform(arr)
    prediction = int(modele_api.predict(arr_scaled)[0])
    proba      = float(modele_api.predict_proba(arr_scaled)[0][prediction])
    label      = target_names[prediction]

    return jsonify({
        "prediction": prediction,
        "proba": round(proba, 4),
        "label": label
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": type(modele_api).__name__})

# Lancer Flask dans un thread + exposer avec ngrok
def run_flask():
    app.run(port=5001, use_reloader=False)

ngrok.set_auth_token("3EzhmBXPh4Pe2dW17QAlD9ttpcS_7Svn8zN6ck5QFL7rq9EUF")

thread = threading.Thread(target=run_flask, daemon=True)
thread.start()
time.sleep(2)

tunnel = ngrok.connect(5001)
public_url = tunnel.public_url

print(f"\n✓ API accessible sur : {public_url}/predict")
print(f"✓ Health check : {public_url}/health")
print("\nTest avec curl (copiez dans un terminal ou une autre cellule) :")
print(
    f"""curl -s -X POST {public_url}/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"features": {X_test[0].round(4).tolist()}}}'"""
)
thread.join()