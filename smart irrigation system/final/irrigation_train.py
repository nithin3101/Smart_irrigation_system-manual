import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pickle

# Load dataset
df = pd.read_csv("balanced_irrigation_dataset.csv")

# Convert temperature range to average value
def convert_range(value):
    if isinstance(value, str) and '-' in value:
        parts = value.split('-')
        return (float(parts[0]) + float(parts[1])) / 2
    try:
        return float(value)
    except:
        return None

df['temperature'] = df['temperature'].apply(convert_range)

# Drop rows with missing values in relevant columns
df.dropna(subset=['CropType', 'SoilMoisture', 'temperature', 'Humidity', 'IrrigationNeeded'], inplace=True)

# Encode 'CropType' column
label_encoder = LabelEncoder()
df['CropType'] = label_encoder.fit_transform(df['CropType'])

# Select features and target
features = ['CropType', 'SoilMoisture', 'temperature', 'Humidity']
X = df[features]
y = df['IrrigationNeeded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save model and encoder
joblib.dump(clf, 'irrigation_classifier.pkl')
with open('irrigation_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("💾 Model and encoder saved successfully.")
