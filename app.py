from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and column names
model = joblib.load("models/heart_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    # Get values from the form
    age = int(request.form["Age"])
    restingbp = int(request.form["RestingBP"])
    cholesterol = int(request.form["Cholesterol"])
    maxhr = int(request.form["MaxHR"])

    # Create input dictionary with all columns set to 0
    input_data = {col: 0 for col in model_columns}

    # Fill numerical values
    if "Age" in input_data:
        input_data["Age"] = age
    if "RestingBP" in input_data:
        input_data["RestingBP"] = restingbp
    if "Cholesterol" in input_data:
        input_data["Cholesterol"] = cholesterol
    if "MaxHR" in input_data:
        input_data["MaxHR"] = maxhr

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        result = "⚠️ High Risk of Heart Disease"
    else:
        result = "✅ Low Risk of Heart Disease"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)