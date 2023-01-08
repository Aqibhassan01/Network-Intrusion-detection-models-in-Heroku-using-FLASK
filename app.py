import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    prediction = model.predict([[20, 9       , 491        , 0        , 0        ,  1.00      , 0.00      , 25, 0.17      , 0.03      ]])

    output = (prediction)

    return render_template('index.html', prediction_text='This is our prediction  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)