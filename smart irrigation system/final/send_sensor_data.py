import requests
import serial
import time  # Add delay between sends

SERIAL_PORT = 'COM9'  # Adjust to your actual port
BAUD_RATE = 9600
SERVER_URL = 'http://localhost:5000/sensor_input'

def send_data(soil, humidity, temp):
    data = {
        'soilMoisture': soil,
        'humidity': humidity,
        'temperature': temp
    }
    try:
        response = requests.post(SERVER_URL, data=data)
        print("📩 Sent:", response.json())
    except Exception as e:
        print("❌ Error sending:", e)

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        print("Listening on", SERIAL_PORT)

        while True:
            line = ser.readline().decode().strip()
            if line:
                print("📡 From Arduino:", line)
                parts = line.split(',')
                if len(parts) == 3:
                    try:
                        soil = float(parts[0])
                        humidity = float(parts[1])
                        temp = float(parts[2])
                        send_data(soil, humidity, temp)
                        time.sleep(2)  # To avoid flooding
                    except:
                        print("⚠️ Invalid float format")
    except Exception as e:
        print("❌ Serial Error:", e)

if __name__ == '__main__':
    main()
