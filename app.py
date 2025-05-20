from flask import Flask, request, jsonify, render_template
import mlflow
import numpy as np
import joblib
import os

app = Flask(__name__)

mlflow.set_tracking_uri("http://127.0.0.1:9090")
MODEL_NAME = "penguins" # Asegúrate de usar el nombre correcto si usaste CSV
MODEL_VERSION = "1"

# Cargar el modelo registrado desde MLflow
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.pyfunc.load_model(model_uri=model_uri)


# Obtener el run_id asociado con la versión del modelo
import mlflow.tracking
client = mlflow.tracking.MlflowClient()
mv = client.get_model_version(MODEL_NAME, MODEL_VERSION)
run_id_scaler = mv.run_id

# Cargar el scaler usando el run_id
scaler = None
#try:
    # local_path = mlflow.artifacts.download_artifacts(run_id=run_id_scaler, artifact_path="scaler")
    #scaler = joblib.load(os.path.join(local_path, "scaler.joblib"))
#except Exception as e:
#    print(f"Error al cargar el escalador: {e}")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not all(key in request.form for key in ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']):
            return jsonify({'error': 'Faltan datos del formulario'}), 400

        bill_length_mm = request.form['bill_length_mm']
        bill_depth_mm = request.form['bill_depth_mm']
        flipper_length_mm = request.form['flipper_length_mm']
        body_mass_g = request.form['body_mass_g']

        features = np.array([[float(bill_length_mm), float(bill_depth_mm), float(flipper_length_mm), float(body_mass_g)]])
        if scaler is not None:
            scaled_features = scaler.transform(features)
            prediction_raw = model.predict(scaled_features).tolist()

            # Convertir la predicción numérica a la etiqueta de la especie (si es necesario)
            species_labels = ['Adelie', 'Chinstrap', 'Gentoo'] # Ajusta según tu codificación
            prediction = species_labels[prediction_raw[0]]
        else:
            prediction_raw = model.predict(features).tolist()
            prediction = str(prediction_raw)
            print("¡Advertencia! Prediciendo sin escalar.")

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)