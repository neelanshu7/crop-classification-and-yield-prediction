from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib  # For loading the ML model

app = Flask(__name__)

# Load your trained ML models
crop_model = joblib.load('ensemble_model.pkl')
yield_model = joblib.load('xgboost_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommendation', methods=['POST'])
def recommend_crop():
    # Get input data from the form
    nitrogen = float(request.form['Nitrogen'])
    phosphorus = float(request.form['Phosporus'])
    potassium = float(request.form['Potassium'])
    temperature = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    # Prepare input for the model
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    
    # Predict crop
    recommended_crop = crop_model.predict(input_data)[0]
    predicted_crop_name = label_encoder.inverse_transform([recommended_crop])[0]
    return f'Recommended Crop: {predicted_crop_name}'

@app.route('/yield.html', methods=['GET'])
def yield_page():
    return render_template('yield.html')

@app.route('/predict', methods=['POST'])
def predict_yield():
    # Get input data from the form
    state = int(request.form['State'])
    season = int(request.form['Season'])
    crop = int(request.form['Crop'])
    area = float(request.form['Area'])

    # Prepare input for the model
    input_data = np.array([[state, season, crop, area]])
    # Predict yield
    predicted_yield = yield_model.predict(input_data)[0]

    return f'Predicted Yield: {predicted_yield:.2f} tonnes'

if __name__ == '__main__':
    app.run(debug=True) 