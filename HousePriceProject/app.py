import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import os


app=Flask(__name__)#starting point of application where the application starts 
model=pickle.load(open('regmodel.pkl','rb'))#Load the model
scaler =pickle.load(open('scaling.pkl','rb'))

@app.route('/check')
def check():
    return str(os.listdir('templates'))

@app.route('/')# Go to my home the '/' Should be says like that
def home():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Home</title>
</head>
<body>
    <h1>Flask is working 🚀</h1>
</body>
</html>"""

@app.route('/predict_api',methods=['POST'])#capture the input and send to the model and model give the output
def predict_api():
    data=request.json['data']#when click the 'predict_api' then it request through the json and capture and stored in the dat variable
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_transformData=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_transformData)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)
