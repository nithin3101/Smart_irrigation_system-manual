import pandas as pd
import joblib

# Load the saved model and encoders
model = joblib.load('water_required_regression_model.pkl')
encoders = joblib.load('regression_crop_encoder.pkl')

# Example input (remove 'Region' from the input data)
input_data = {
    'Crop': ['Paddy'],
    'Acreage': [5],                 # in acres
    'CropDays': [70],
    'Temperature': [34],
    'SoilMoisture': [12],
    'Humidity': [55],
    'Evapotranspiration': [5.2],
    'CropCoefficient': [1.05]
}

input_df = pd.DataFrame(input_data)

# Encode only 'Crop' (Region is not used anymore)
input_df['Crop'] = encoders['Crop'].transform(input_df['Crop'])

# Predict
predicted_water = model.predict(input_df)[0]
print(f"💧 Predicted Water Requirement: {predicted_water:.2f} liters (or mm)")

