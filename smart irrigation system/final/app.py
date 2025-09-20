from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
from collections import deque

app = Flask(__name__)
CORS(app)

# Load model and encoders
model = joblib.load('water_required_regression_model.pkl')
encoders = joblib.load('regression_crop_encoder.pkl')

# Global storage for sensor data
sensor_data = {
    'soilMoisture': None,
    'temperature': None,
    'humidity': None
}

# For rendering live data
latest_data = {
    'temperature': None,
    'humidity': None,
    'soil': None
}

# Store last 5 entries (deque = efficient for FIFO)
history = deque(maxlen=5)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_irrigation')
def predict_irrigation():
    return render_template('predict_irrigation.html', data=latest_data)

@app.route('/manual_input', methods=['POST'])
def manual_input():
    """Enter soil, temperature, humidity manually"""
    try:
        soil = float(request.form['soilMoisture'])
        humidity = float(request.form['humidity'])
        temperature = float(request.form['temperature'])

        # Update globals
        sensor_data['soilMoisture'] = soil
        sensor_data['humidity'] = humidity
        sensor_data['temperature'] = temperature

        latest_data['soil'] = soil
        latest_data['humidity'] = humidity
        latest_data['temperature'] = temperature

        # Add to history
        history.append({
            'soilMoisture': soil,
            'humidity': humidity,
            'temperature': temperature
        })

        return jsonify({'message': 'Manual data entered successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/analytics', methods=['GET'])
def analytics():
    """Analytics on last 5 entries"""
    if not history:
        return jsonify({'error': 'No data available'}), 400

    df = pd.DataFrame(list(history))
    averages = df.mean().to_dict()

    # Simple trend detection (compare last vs first)
    trend = {}
    for col in df.columns:
        if df[col].iloc[-1] > df[col].iloc[0]:
            trend[col] = "Increasing"
        elif df[col].iloc[-1] < df[col].iloc[0]:
            trend[col] = "Decreasing"
        else:
            trend[col] = "Steady"

    return jsonify({
        'history': list(history),
        'averages': averages,
        'trend': trend
    })

@app.route('/get_sensor_data', methods=['GET'])
def get_sensor_data():
    return jsonify(sensor_data)

@app.route('/live_data', methods=['GET'])
def live_data():
    return jsonify(latest_data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        crop = request.form['crop']
        acreage = float(request.form['acreage'])

        soil = sensor_data['soilMoisture']
        humidity = sensor_data['humidity']
        temp = sensor_data['temperature']

        if soil is None or humidity is None or temp is None:
            return jsonify({'error': 'Sensor data not available'}), 400

        suggestion = "Irrigate" if soil < 30 else "Do not irrigate"
        prediction = None

        if suggestion == "Irrigate":
            input_df = pd.DataFrame({
                'Crop': [crop],
                'Acreage': [acreage],
                'CropDays': [70],
                'Temperature': [temp],
                'SoilMoisture': [soil],
                'Humidity': [humidity],
                'Evapotranspiration': [5.2],
                'CropCoefficient': [1.05]
            })

            input_df['Crop'] = encoders['Crop'].transform(input_df['Crop'])
            prediction = round(model.predict(input_df)[0], 2)

        return render_template('predict_irrigation.html',
                               soilMoisture=soil,
                               temperature=temp,
                               humidity=humidity,
                               suggestion=suggestion,
                               prediction=prediction)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
