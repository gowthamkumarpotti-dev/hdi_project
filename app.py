from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def classify_hdi(score):
    if score >= 0.8:
        return 'Very High', '#27ae60', '🌟'
    elif score >= 0.7:
        return 'High', '#2980b9', '✅'
    elif score >= 0.55:
        return 'Medium', '#f39c12', '⚠️'
    else:
        return 'Low', '#e74c3c', '❗'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        try:
            life_exp  = float(request.form['life_expectancy'])
            schooling = float(request.form['schooling'])
            gni       = float(request.form['gni'])

            features = np.array([[life_exp, schooling, gni]])
            hdi_score = model.predict(features)[0]
            hdi_score = round(float(np.clip(hdi_score, 0, 1)), 4)

            category, color, icon = classify_hdi(hdi_score)
            result = {
                'score': hdi_score,
                'category': category,
                'color': color,
                'icon': icon,
                'life_exp': life_exp,
                'schooling': schooling,
                'gni': gni
            }
        except Exception as e:
            result = {'error': str(e)}
    return render_template('predict.html', result=result)

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

if __name__ == '__main__':
    app.run(debug=True)
