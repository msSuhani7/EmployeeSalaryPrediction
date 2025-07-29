from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model (make sure model.pkl is in same folder)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    workclass = int(request.form['workclass'])
    marital = int(request.form['marital'])
    occupation = int(request.form['occupation'])
    education = int(request.form['education'])
    hours = int(request.form['hours'])
    capital_gain = int(request.form['capital_gain'])
    capital_loss = int(request.form['capital_loss'])

    # Input order must match your training
    features = np.array([[age, workclass, marital, occupation, education, hours, capital_gain, capital_loss, gender]])
    prediction = model.predict(features)[0]

    result = "Income >50K" if prediction == 1 else "Income <=50K"
    return render_template('form.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
