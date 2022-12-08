import flask
from flask import Flask, render_template, request
import pickle as pk
import numpy as np
import pandas as pd
app=Flask(__name__)

loanModel = pk.load(open("model.pkl", "rb")) 
loanLabelEncoder = pk.load(open("label_encoder_map.pkl", "rb"))

@app.route('/')
def index():
    return flask.render_template('index.html')
 
@app.route('/predict', methods = ['POST'])
def result():
    valuesDict = request.form.to_dict()
    for key in valuesDict:
        if(valuesDict[key] == ""): return render_template('predict.html', prediction="ERROR: Please enter all inputs")
    valuesDict["LoanAmount"] = str(int(int(valuesDict["LoanAmount"])/1000))
    
    keys = list(request.form.to_dict().keys())
    values = np.array(list(valuesDict.values())).reshape(1,11)
    df = pd.DataFrame(values, columns = keys)

    X = df[keys] 

    probability = loanModel.predict_proba(df[X.columns])[:,1] 
    probability = "{:.2%}".format(probability[0])
    print("PROB: ", probability[0])

    return render_template('predict.html', prediction="Probability of Loan Being Accepted: " + probability)

if __name__ == '__main__':
    app.run(debug=True, port=9090)
   

