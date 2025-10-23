from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('svm_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    inputs = [float(x) for x in request.form.values()]
    input_data = np.array(inputs).reshape(1, -1)
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return render_template('result.html', prediction_text=f"The person is {result}")

if __name__ == "__main__":
    app.run(debug=True)
