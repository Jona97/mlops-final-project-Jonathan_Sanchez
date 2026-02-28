import os
import pandas as pd
import mlflow.sklearn

from flask import Flask, request, jsonify, render_template

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../src
PROJECT_ROOT = os.path.dirname(BASE_DIR)                   # root
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "best_model")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# -------------------------------
# APP FLASK
# -------------------------------
app = Flask(__name__, template_folder=TEMPLATES_DIR)

# -------------------------------
# CARGAR MODELO
# -------------------------------
print(f"Cargando modelo desde: {MODEL_DIR}")
model = mlflow.sklearn.load_model(MODEL_DIR)
print("Modelo cargado correctamente")

# -------------------------------
# HEALTH CHECK
# -------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# -------------------------------
# FORMULARIO + PREDICCION
# -------------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():

    # 👉 Si es GET → mostrar formulario
    if request.method == "GET":
        return render_template("predict.html")

    # 👉 Si es POST → leer datos del formulario
    age = int(request.form["age"])
    sex = request.form["sex"]
    bmi = float(request.form["bmi"])
    children = int(request.form["children"])
    smoker = request.form["smoker"]
    region = request.form["region"]

    X = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    pred = model.predict(X)[0]

    return render_template(
        "predict.html",
        prediction=round(float(pred), 2)
    )

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    print(app.url_map)
    app.run(host="0.0.0.0", port=5000, debug=True)