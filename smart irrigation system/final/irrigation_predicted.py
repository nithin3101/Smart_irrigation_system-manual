import pandas as pd
import joblib

# Load the trained model and encoder
model = joblib.load('irrigation_classifier.pkl')
encoder = joblib.load('irrigation_encoder.pkl')  # this must match what you used during training


# 🔸 Provide prediction input matching training columns
input_data = {
    'CropType': ['Wheat'],
    'SoilMoisture': [10],        # very low moisture
    'temperature': [36],         # hot
    'Humidity': [40]             # low humidity
}
# Convert to DataFrame
input_df = pd.DataFrame(input_data)

# Encode the CropType column just like in training
input_df['CropType'] = encoder.transform(input_df['CropType'])

# Predict irrigation need
prediction = model.predict(input_df)[0]

# Print result
if prediction == 1:
    print("✅ Irrigation Needed")
else:
    print("🚫 No Irrigation Needed")
