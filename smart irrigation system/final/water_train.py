import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# 🔹 Load your dataset
data = pd.read_csv('synthetic_crop_water_requirement_dataset.csv')  # Replace with your actual file

# 🔹 Encode categorical features
crop_encoder = LabelEncoder()
# You no longer need to encode 'Region' if it's not used as an input feature

data['Crop'] = crop_encoder.fit_transform(data['Crop'])

# 🔹 Save encoder for 'Crop' (only)
encoders = {
    'Crop': crop_encoder
}

# 🔹 Define input features and target (Remove 'Region' from features)
features = ['Crop', 'Acreage', 'CropDays', 'Temperature',
            'SoilMoisture', 'Humidity', 'Evapotranspiration', 'CropCoefficient']
target = 'WaterRequirement'

X = data[features]
y = data[target]

# 🔹 Train-test split (optional, for validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error, r2_score

# 🔹 Predict on the test set
y_pred = model.predict(X_test)

# 🔹 Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Mean Squared Error (MSE): {mse:.2f}")
print(f"📈 R² Score: {r2:.2f}")


# 🔹 Save model and encoder
joblib.dump(model, 'water_required_regression_model.pkl')
joblib.dump(encoders, 'regression_crop_encoder.pkl')

print("✅ Model and encoders saved successfully!")
