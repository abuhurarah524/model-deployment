from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# create app and load the trained model
app = Flask(__name__)
model = pickle.load(open('Linear_MI_model.pkl', 'rb'))

#Route to handle HOME
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

standard_to = StandardScaler()

#Route to handle Predicted Result
@app.route("/", methods=['POST'])
def predict():
    if request.method == 'POST':  
        CreditScore=int(request.form['CreditScore'])
        DTI=int(request.form['DTI'])
        MaturityMonth=int(request.form['MaturityMonth']) 
        FirstPaymentMonth=int(request.form['FirstPaymentMonth'])
        LTV=int(request.form['LTV'])
        OCLTV=int(request.form['OCLTV'])
        OrigUPB=int(request.form['OrigUPB'])
        OrigInterestRate=float(request.form['OrigInterestRate'])
        PostalCode=int(request.form['PostalCode'])
        MIP=int(request.form['MIP'])      
        NumBorrowers=int(request.form['NumBorrowers'])
        PropertyState=request.form['PropertyState']
        LoanPurpose=request.form['LoanPurpose']
        Channel=request.form['Channel']
        PropertyType=request.form['PropertyType']
        data = {
             'PropertyState': [PropertyState],
             'Channel': [Channel],
            'LoanPurpose': [LoanPurpose],
            'PropertyType': [PropertyType]
            }
        df = pd.DataFrame(data)
        label_encoder = LabelEncoder()
        # Apply label encoding to each categorical column
        PropertyState = label_encoder.fit_transform([PropertyState])[0]
        Channel = label_encoder.fit_transform([Channel])[0]
        LoanPurpose = label_encoder.fit_transform([LoanPurpose])[0]
        PropertyType = label_encoder.fit_transform([PropertyType])[0]
        input_data = [[DTI, MaturityMonth, FirstPaymentMonth, OCLTV, OrigInterestRate, LTV, MIP, NumBorrowers,
                               OrigUPB, PropertyState, LoanPurpose, Channel, PostalCode, PropertyType, CreditScore]]
                            
        # Perform the prediction
        prediction = model.predict(input_data)
        output=round(prediction[0],5)
        scaler = MinMaxScaler()
        ouput = scaler.fit_transform([[output]])
        if output<0.0000000000000000:
            return render_template('index.html',prediction_text="the prepayment risk is 0")
        else:
            return render_template('index.html',prediction_text="The Prepayment Risk is {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
