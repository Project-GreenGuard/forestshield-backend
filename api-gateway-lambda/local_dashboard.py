from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

latest_data = {
    'temperature': None,
    'sensor_id': None,
    'location': None
}

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Temperature Dashboard</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background-color: black;
            color: white;
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .sensor-info {
            font-size: 30px;
            color: #888;
            margin-bottom: 20px;
        }
        .temperature {
            font-size: 120px;
            font-weight: bold;
            color: #00ff00;
        }
        .temperature.high {
            color: #ff0000;
        }
        .location {
            font-size: 40px;
            color: #888;
            margin-top: 20px;
        }
        .fire-warning {
            display: none;
            background-color: #ff0000;
            color: white;
            padding: 30px 60px;
            font-size: 50px;
            font-weight: bold;
            border-radius: 10px;
            margin-top: 40px;
            animation: blink 1s infinite;
            text-align: center;
        }
        .fire-warning.show {
            display: block;
        }
        @keyframes blink {
            0%, 50%, 100% { opacity: 1; }
            25%, 75% { opacity: 0.5; }
        }
    </style>
    <script>
        function updateTemp() {
            fetch('/api/temperature')
                .then(response => response.json())
                .then(data => {
                    if (data.temperature !== null) {
                        document.getElementById('temp').textContent = data.temperature + '°C';
                        
                        // Check for fire warning
                        const tempElement = document.getElementById('temp');
                        const warningElement = document.getElementById('fire-warning');
                        
                        if (data.temperature > 35) {
                            tempElement.classList.add('high');
                            warningElement.classList.add('show');
                        } else {
                            tempElement.classList.remove('high');
                            warningElement.classList.remove('show');
                        }
                    }
                    if (data.sensor_id !== null) {
                        document.getElementById('sensor').textContent = 'Sensor: ' + data.sensor_id;
                    }
                    if (data.location !== null) {
                        document.getElementById('location').textContent = 'Location: ' + data.location;
                    }
                });
        }
        setInterval(updateTemp, 2000);
        updateTemp();
    </script>
</head>
<body>
    <div class="sensor-info" id="sensor">Sensor: --</div>
    <div class="temperature" id="temp">--°C</div>
    <div class="location" id="location">Location: --</div>
    <div class="fire-warning" id="fire-warning">⚠️ FIRE WARNING ⚠️<br>High Temperature Detected!</div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/temperature', methods=['POST'])
def receive_temperature():
    global latest_data
    data = request.get_json()
    latest_data['temperature'] = data.get('temperature')
    latest_data['sensor_id'] = data.get('sensor_id', 'Unknown')
    latest_data['location'] = data.get('location', 'Unknown')
    
    print(f"Received from {latest_data['sensor_id']} ({latest_data['location']}): {latest_data['temperature']}°C")
    return jsonify({'success': True})

@app.route('/api/temperature', methods=['GET'])
def get_temperature():
    return jsonify(latest_data)

if __name__ == '__main__':
    print("Dashboard running at http://192.168.2.164:5000")
    app.run(host='192.168.2.164', port=5000)
