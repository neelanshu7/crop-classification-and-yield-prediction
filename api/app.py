from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib  # For loading the ML model

app = Flask(__name__)
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

crop_model = joblib.load(os.path.join(BASE_DIR, 'ensemble_model.pkl'))
yield_model = joblib.load(os.path.join(BASE_DIR, 'xgboost_model.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))   # Regression
scaler1 = joblib.load(os.path.join(BASE_DIR, 'scaler1.pkl')) # Classification

# # Load your trained ML models
# crop_model = joblib.load('ensemble_model.pkl')
# yield_model = joblib.load('xgboost_model.pkl')
# label_encoder = joblib.load('label_encoder.pkl')
# scaler=joblib.load('scaler.pkl')    # Regression
# scaler1=joblib.load('scaler1.pkl')  # Classification

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommendation.html', methods=['POST'])
def recommend_crop():
    # Get input data from the form
    nitrogen = int(request.form['Nitrogen'])
    potassium = int(request.form['Potassium'])
    phosphorus = int(request.form['Phosporus'])
    temperature = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    # Prepare input for the model
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    scaled_input = scaler1.transform(input_data)
    # Predict crop
    recommended_crop = crop_model.predict(scaled_input)[0]
    predicted_crop_name = label_encoder.inverse_transform([recommended_crop])[0]
    # return f'Recommended Crop: {predicted_crop_name}'
    return render_template('recommendation.html', nitrogen=nitrogen, phosphorus=phosphorus, potassium=potassium,
                           temperature=temperature, humidity=humidity, ph=ph, rainfall=rainfall,
                           recommended_crop=predicted_crop_name)

@app.route('/yield.html', methods=['GET'])
def yield_page():
    return render_template('yield.html')

@app.route('/predict.html', methods=['POST'])
def predict_yield():
    # Get input data from the form
    state = int(request.form['State'])
    season = int(request.form['Season'])
    crop = int(request.form['Crop'])
    area = float(request.form['Area'])

    # Prepare input for the model
    input_data = np.array([[state, season, crop, area]])
    scaled_input = scaler.transform(input_data)
    # Predict yield
    predicted_yield = yield_model.predict(scaled_input)[0]

    # return f'Predicted Yield: {predicted_yield:.2f} tonnes'
    return render_template('predict.html', state=state, season=season, crop=crop, area=area, predicted_yield=predicted_yield)

# if __name__ == '__main__':
#     app.run(debug=True) 