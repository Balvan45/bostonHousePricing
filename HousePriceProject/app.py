import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

app=Flask(__name__,template_folder=os.path.join(base_dir, 'templates'),
            static_folder=os.path.join(base_dir, 'static'))#starting point of application where the application starts 
model=pickle.load(open('regmodel.pkl','rb'))#Load the model
scaler =pickle.load(open('scaling.pkl','rb'))

@app.route('/check')
def check():
    return str(os.listdir('templates'))

@app.route('/')# Go to my home the '/' Should be says like that
def home():
    return render_template("home.html")

@app.route('/predict_api',methods=['POST'])#capture the input and send to the model and model give the output
def predict_api():
    data=request.json['data']#when click the 'predict_api' then it request through the json and capture and stored in the data variable
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_transformData=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_transformData)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]#Capture the values when we fill the values in the form 
    final_input = scaler.transform(np.array(data).reshape(1,-1))

    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The predicted House price is{} ".format(output))

if __name__=="__main__":
    app.run(debug=True,port=8080)
