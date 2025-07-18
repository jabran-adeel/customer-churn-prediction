# Import necessary libraries
import pickle
import numpy as np
import pandas as pd

# 1. Load the model and transformers
with open("model/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# 2. Define manual input dictionary
input_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80.5,
    "TotalCharges": 400.0
}

# 3. Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# 4. Encode categorical features
for col in input_df.columns:
    if col in label_encoders:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

# 5. Scale numeric features
scaled_input = scaler.transform(input_df)

# 6. Predict
prediction = model.predict(scaled_input)[0]
probability = model.predict_proba(scaled_input)[0][1]

print("\n📊 Prediction Result:")
print("Churn" if prediction == 1 else "Not Churn")
print(f"Probability of churn: {probability:.2f}")