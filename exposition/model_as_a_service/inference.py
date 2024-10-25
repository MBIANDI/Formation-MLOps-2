from flask import Flask, jsonify
from formation_indus_ds_avancee.feature_engineering import prepare_features
from formation_indus_ds_avancee.train_and_predict import predict
from config import MODEL_PATH, INFERENCE_DATA

app = Flask(__name__)


@app.route('/health')
def health():
    return jsonify({
        "status": "ok"
    })


@app.route('/predict')
def predict_endpoint():
    
    features = prepare_features(data=INFERENCE_DATA, training_mode=False)
    features = predict(features, model_path= MODEL_PATH)

    return jsonify(features['predictions'])
