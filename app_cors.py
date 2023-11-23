from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS, cross_origin

import joblib
import numpy as np

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

#Rota para a página inicial
@app.route('/')
def formulario():
    #Renderiza o template HTML chamado 'formulario.html'
    return render_template('formulario.html')

#Rota para previsão
@app.route('/prever_diabetes', methods=['POST'])
def prever_diabetes():
    
    data = request.json
    
    #Extração das features
    features = [data['preg'], data['plas'], data['pres'], data['skin'], data['test'], data['mass'], data['pedi'], data['age']]
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    #Redimensionar as features usando o scaler treinado
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)  # Use features_scaled here
    
    #Mapeamento dos resultados das classes
    class_mapping = {0: 'Sem diabetes', 1: 'Com diabetes'}
    #Devolve a previsão
    return jsonify({'prediction': class_mapping[prediction[0]]})

if __name__ == '__main__':
    app.run(debug=True)
