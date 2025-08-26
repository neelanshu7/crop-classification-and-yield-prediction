# 🌾 Crop Classification and Yield Prediction 

### 🔗 Live Website: [Plantalytics](https://plantalytics.onrender.com)

### 🔗 Live Website: [Plantalytics](https://plantalytics.onrender.com)

---

## 🚀 Overview

At the heart of modern agriculture lies the need for **innovation**, **sustainability**, and **precision**.  
**Plantalytics** empowers farmers and agricultural stakeholders with cutting-edge solutions for **Crop Recommendation** and **Yield Prediction**, harnessing the power of **Machine Learning** and **data analytics** to transform traditional farming into smart agriculture.

---

## 🌱 Crop Recommendation: Smart Choices for Better Farming

Our **Crop Recommendation System** helps farmers make **informed decisions** about which crops to cultivate, tailored to their **local environmental and economic conditions**.  

🔍 **Key Features:**
- Personalized recommendations using soil data, temperature, rainfall, and pH levels
- Integration of machine learning algorithms for adaptive decision-making
- Designed for maximum productivity and profitability

---

## 🌾 Yield Prediction: Forecasting the Future of Harvests

**Yield Prediction** enables farmers to anticipate harvest outcomes with accuracy—essential for resource planning and market readiness.  

📊 **Technologies Used:**
- Machine Learning models trained on historical crop yield data
- Predictive insights to improve decision-making and reduce risk

---

## 📁 About This Repository

This repository contains:
- 📓 Jupyter Notebooks for model training and evaluation (located in the `notebook/` directory)
- 📊 Sample datasets (`.csv` files) for training and testing
- 🌐 Flask-based web application for user interaction

---

## 💻 How to Run the Project Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/plantalytics.git
   git clone https://github.com/neelanshu7/crop-classification-and-yield-prediction.git
   cd crop-classification-and-yield-prediction

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Run the Flask server:**
   ```bash
   flask run --host=0.0.0.0

4. Access the application:
   Open your browser and navigate to **http://localhost:5000/**

## 📌 Project Structure
   ```php
Crop-Classification-Yield-Prediction/
crop-classification-and-yield-prediction/
│
├── 📂 .ipynb_checkpoints/      // Auto-saved Jupyter checkpoints
├── 📂 __pycache__/             // Compiled Python files
│
├── 📂 notebook/               // Jupyter Notebooks and datasets
│   ├── crop_prediction.ipynb
│   ├── yield_prediction.ipynb
│   └── indiancrop.csv
│   └── crop_production.csv
│   └── filtered_crop_production.csv
│   └── filtered_crop_production_imputed.csv
│
├── 📂 static/                 // Static assets (CSS)
│   ├── style.css
│
├── 📂 templates/              // Flask HTML templates
│   ├── index.html
│   └── predict.html
│   └── recommendation.html
│   └── yield.html
│
├── 📄 app.py                  // Main Flask application
├── 📄 ensemble_model.pkl      // Ensemble ML model
├── 📄 label_encoder.pkl       // Label encoder for crop and yield
├── 📄 scaler.pkl              // Feature scaler for crop classification
├── 📄 scaler1.pkl             // Feature scaler for yield prediction
├── 📄 xgboost_model.pkl       // XGBoost ML model
├── 📄 requirements.txt        // Python dependencies
├── 📄 README.md               // Project  overview & setup
