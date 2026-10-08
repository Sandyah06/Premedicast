from flask import Flask, render_template_string, request, send_file
import pandas as pd
from datetime import datetime, timedelta
import requests
import joblib
import io
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import traceback
import numpy as np

# Configure matplotlib
plt.switch_backend('Agg')
plt.style.use('seaborn-v0_8')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

app = Flask(__name__)

# Load dataset and model
try:
    df = pd.read_csv('final_merged_dataset.csv', parse_dates=['record_date'])
    df['record_date'] = pd.to_datetime(df['record_date'], errors='coerce')
    model = joblib.load("absenteeism_model.pkl")
except Exception as e:
    print(f"Error loading data/model: {str(e)}")
    df = pd.DataFrame()
    model = None

API_KEY = "a710553d9d99f30326a82d432265cf36"

def get_coordinates_from_pincode(pincode):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {'q': f'{pincode}, Karnataka, India', 'format': 'json', 'limit': 1}
        response = requests.get(url, params=params, headers={"User-Agent": "weather-check"})
        response.raise_for_status()
        data = response.json()
        if data:
            location_name = data[0]['display_name'].split(',')[0]
            return float(data[0]['lat']), float(data[0]['lon']), location_name
    except Exception as e:
        print(f"⚠️ Error fetching coordinates: {e}")
    return 12.9716, 77.5946, "Bengaluru"

def verify_pincode_in_karnataka(pincode):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {'postalcode': pincode, 'country': 'India', 'format': 'json', 'limit': 1}
        response = requests.get(url, params=params, headers={"User-Agent": "pincode-verifier"})
        response.raise_for_status()
        data = response.json()
        if data:
            return 'karnataka' in data[0]['display_name'].lower()
    except Exception as e:
        print(f"⚠️ Pincode verification error: {e}")
    return False

def get_realtime_weather(pincode):
    try:
        lat, lon, _ = get_coordinates_from_pincode(pincode)
        
        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        weather_params = {'lat': lat, 'lon': lon, 'units': 'metric', 'appid': API_KEY}
        weather_response = requests.get(weather_url, params=weather_params)
        weather_response.raise_for_status()
        weather = weather_response.json()
        
        aq_url = "http://api.openweathermap.org/data/2.5/air_pollution"
        aq_response = requests.get(aq_url, params=weather_params)
        aq_response.raise_for_status()
        air = aq_response.json()

        return {
            'temperature': weather['main']['temp'],
            'humidity': weather['main']['humidity'],
            'wind_speed': weather['wind']['speed'],
            'conditions': weather['weather'][0]['description'],
            'air_quality': air['list'][0]['main']['aqi']
        }
    except Exception as e:
        print(f"⚠️ Weather API error: {e}")
        return None

def predict_weather_illness_and_absenteeism(pincode, input_date_str):
    try:
        if not verify_pincode_in_karnataka(pincode):
            return "❌ Error: Invalid Karnataka pincode", {
                "location": "Unknown",
                "prediction_date": "",
                "risk_level": "Unknown",
                "risk_color": "#cccccc",
                "weather_data": {},
                "illness_reasons": "",
                "predicted_value": 0,
                "flu_risk": 0,
                "er_rate": 0
            }

        input_date = pd.to_datetime(input_date_str)
        target_day = input_date + timedelta(days=14)
        _, _, location_name = get_coordinates_from_pincode(pincode)

        # Initialize all variables
        weather_data = {
            'current': {},
            'averages': {},
            'combined': {}
        }
        illness = set()
        absenteeism_boost = 0
        flu_risk_percent = 0.0
        flu_er_rate = 0.0
        er_visit_rate = 0
        risk_level = "Low"
        ml_prediction = 0
        combined_prediction = 0
        reasons = "No major illness risk"
        risk_color = "#00cc00"
        base_absenteeism = 10  # Default value

        # Get real-time weather
        rt = get_realtime_weather(pincode)
        if rt:
            weather_data['current'] = {
                'temperature': rt['temperature'],
                'air_quality': rt['air_quality'],
                'humidity': rt['humidity'],
                'wind_speed': rt['wind_speed'],
                'conditions': rt['conditions']
            }
            print(f"📍 Real-time → Temp: {rt['temperature']}°C, AQI: {rt['air_quality']}, "
                  f"Humidity: {rt['humidity']}%, Wind: {rt['wind_speed']} m/s, "
                  f"Condition: {rt['conditions']}")

        # Process historical data
        df_filtered = df[df['pincode'] == int(pincode)].copy()
        df_filtered['month_day'] = df_filtered['record_date'].dt.strftime('%m-%d')
        window_days = [(target_day + timedelta(days=offset)).strftime('%m-%d') for offset in range(-15, 16)]
        past_days = df_filtered[df_filtered['month_day'].isin(window_days)]

        if not past_days.empty:
            for col in ['temperature', 'air_quality', 'humidity', 'wind_speed',
                       'Chronic_Absenteeism', 'FLU_CASES_LAST_MONTH', 'EMERGENCY_SERVICES']:
                past_days[col] = pd.to_numeric(past_days[col], errors='coerce')

            weather_data['averages'] = {
                'temperature': past_days['temperature'].mean(),
                'air_quality': past_days['air_quality'].mean(),
                'humidity': past_days['humidity'].mean(),
                'wind_speed': past_days['wind_speed'].mean()
            }

            if rt:
                weather_data['combined'] = {
                    'temperature': (weather_data['averages']['temperature'] + rt['temperature']) / 2,
                    'air_quality': (weather_data['averages']['air_quality'] + rt['air_quality']) / 2,
                    'humidity': (weather_data['averages']['humidity'] + rt['humidity']) / 2,
                    'wind_speed': (weather_data['averages']['wind_speed'] + rt['wind_speed']) / 2,
                    'conditions': rt['conditions']
                }
            else:
                weather_data['combined'] = weather_data['averages'].copy()
                weather_data['combined']['conditions'] = past_days['conditions'].mode()[0] if not past_days['conditions'].mode().empty else "Unknown"

            # Calculate metrics
            avg_flu_cases = past_days['FLU_CASES_LAST_MONTH'].mean()
            avg_er_visits = past_days['EMERGENCY_SERVICES'].mean()
            safe_flu_cases = avg_flu_cases if pd.notna(avg_flu_cases) else 0
            safe_er_visits = avg_er_visits if pd.notna(avg_er_visits) else 0

            flu_risk_percent = round(min((safe_flu_cases * 0.8 + safe_er_visits * 1.2), 100), 2)
            flu_er_rate = round(min((safe_flu_cases * 0.3 + safe_er_visits * 0.7), 100), 2)
            base_absenteeism = past_days['Chronic_Absenteeism'].mean()

            if flu_risk_percent >= 10:
                illness.update(['Flu'])
                absenteeism_boost += 5

        # Illness and risk calculation
        combined_temp = weather_data['combined']['temperature']
        combined_humidity = weather_data['combined']['humidity']
        combined_air = weather_data['combined']['air_quality']
        combined_wind = weather_data['combined']['wind_speed']
        combined_condition = weather_data['combined']['conditions'].lower()

        if combined_temp < 20:
            illness.update(['Cough', 'Fever'])
            absenteeism_boost += 3
        if combined_temp > 35 and combined_humidity < 30:
            illness.update(['Heatstroke', 'Dehydration'])
            absenteeism_boost += 4
        if combined_humidity > 80 or 'rain' in combined_condition:
            illness.update(['Cold', 'Fungal Infections'])
            absenteeism_boost += 3
        if combined_air > 120 or 'dust' in combined_condition:
            illness.update(['Dust Allergy', 'Asthma'])
            absenteeism_boost += 4
        if combined_wind > 15:
            illness.update(['Eye Irritation', 'Breathing Issues'])
            absenteeism_boost += 2

        predicted_absenteeism = min(base_absenteeism + absenteeism_boost, 100)
        reasons = " / ".join(illness) if illness else "No major illness risk"

        if model and rt:
            try:
                air_quality_encoded = int(rt['air_quality']) - 1
                model_input = [[rt['temperature'], air_quality_encoded, rt['humidity']]]
                ml_prediction = model.predict(model_input)[0]
                combined_prediction = (predicted_absenteeism + ml_prediction) / 2
            except Exception as e:
                print(f"Model prediction error: {e}")
                combined_prediction = predicted_absenteeism
        else:
            combined_prediction = predicted_absenteeism

        er_visit_rate = (absenteeism_boost / 10) * 10
        
        # Determine risk level
        if combined_prediction > 70:
            risk_level = "Very High"
            risk_color = "#ff0000"
        elif combined_prediction > 50:
            risk_level = "High"
            risk_color = "#ff6b00"
        elif combined_prediction > 30:
            risk_level = "Moderate"
            risk_color = "#ffcc00"
        else:
            risk_level = "Low"
            risk_color = "#00cc00"

        weather_desc = (f"{weather_data['combined']['conditions']}, "
                       f"{weather_data['combined']['temperature']:.1f}°C, "
                       f"Humidity: {weather_data['combined']['humidity']:.1f}%, "
                       f"Wind: {weather_data['combined']['wind_speed']:.1f} m/s, "
                       f"AQI: {weather_data['combined']['air_quality']:.1f}")

        return (f"""
📍 Location: {location_name}
📅 Prediction Date: {target_day.date()}
📊 Absenteeism Risk Level: {risk_level}
📈 Final Absenteeism Prediction: {combined_prediction:.2f}%
🧠 ML Model Predicted Absenteeism: {ml_prediction:.2f}%
🏥 ER Visit Rate: {er_visit_rate:.2f}%
🦠 Flu Risk: {flu_risk_percent:.2f}%
🚑 Flu-related ER Visit Rate: {flu_er_rate:.2f}%
🌡️ Weather: {weather_desc}
🔍 Illness Reasons: {reasons}
""", {
            "location": location_name,
            "prediction_date": target_day.date(),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "weather_data": weather_data,
            "weather_desc": weather_desc,
            "illness_reasons": reasons,
            "predicted_value": combined_prediction,
            "flu_risk": flu_risk_percent,
            "er_rate": er_visit_rate
        })

    except Exception as e:
        print(f"Prediction error: {traceback.format_exc()}")
        return f"❌ Error in prediction: {str(e)}", {
            "location": "Unknown",
            "prediction_date": "",
            "risk_level": "Error",
            "risk_color": "#cccccc",
            "weather_data": {},
            "weather_desc": "",
            "illness_reasons": "",
            "predicted_value": 0,
            "flu_risk": 0,
            "er_rate": 0
        }

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Absenteeism Predictor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #f5f7fa; }
        .container { display: flex; min-height: 100vh; }
        
        .sidebar { 
            width: 320px; background: #2c3e50; 
            padding: 30px; color: white;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            position: fixed;
            height: 100%;
        }
        .main-content { 
            margin-left: 320px; padding: 40px;
            flex: 1; background: white;
        }
        
        input, button { 
            padding: 12px; margin: 10px 0; 
            border: none; border-radius: 5px;
            width: 100%;
        }
        button { 
            background: #3498db; color: white; 
            cursor: pointer; transition: background 0.3s;
            font-weight: bold;
        }
        button:hover { background: #2980b9; }
        
        .prediction-box { 
            background: #f8f9fa; padding: 25px;
            border-radius: 8px; margin-bottom: 30px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .weather-container {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin: 20px 0;
        }
        .weather-card {
            flex: 1;
            min-width: 200px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .weather-title {
            font-weight: bold;
            color: #7f8c8d;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        .weather-icon {
            margin-right: 8px;
            font-size: 18px;
        }
        .weather-value {
            font-size: 20px;
            font-weight: bold;
        }
        .risk-indicator {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
            background-color: {{ graph_data.risk_color }};
            color: white;
        }
        pre { 
            background: #f0f0f0; padding: 15px;
            border-radius: 5px; white-space: pre-wrap;
            font-family: 'Courier New', monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h1>🌤️ Absenteeism Predictor</h1>
            <p>Predict school absenteeism in Karnataka</p>
            <form method="post">
                <input type="number" name="pincode" placeholder="🏷️ Enter Pincode" required>
                <input type="date" name="date" required>
                <button type="submit">📊 Predict</button>
            </form>
        </div>

        <div class="main-content">
            {% if output %}
                {% if output.startswith('❌') %}
                    <div style="color: #e74c3c; padding: 15px; background: #fdd; border-radius: 5px;">
                        {{ output }}
                    </div>
                {% else %}
                    <h2>📍 {{ graph_data.location }}</h2>
                    <h3>📅 {{ graph_data.prediction_date }}</h3>
                    
                    <div class="weather-container">
                        <div class="weather-card">
                            <div class="weather-title">
                                <span class="weather-icon">🌡️</span> Temperature
                            </div>
                            <div class="weather-value">
                                {{ "%.1f"|format(graph_data.weather_data.combined.temperature) }}°C
                            </div>
                        </div>
                        <div class="weather-card">
                            <div class="weather-title">
                                <span class="weather-icon">💨</span> Air Quality
                            </div>
                            <div class="weather-value">
                                {{ "%.1f"|format(graph_data.weather_data.combined.air_quality) }} AQI
                            </div>
                        </div>
                        <div class="weather-card">
                            <div class="weather-title">
                                <span class="weather-icon">💧</span> Humidity
                            </div>
                            <div class="weather-value">
                                {{ "%.1f"|format(graph_data.weather_data.combined.humidity) }}%
                            </div>
                        </div>
                        <div class="weather-card">
                            <div class="weather-title">
                                <span class="weather-icon">🌬️</span> Wind Speed
                            </div>
                            <div class="weather-value">
                                {{ "%.1f"|format(graph_data.weather_data.combined.wind_speed) }} m/s
                            </div>
                        </div>
                    </div>
                    
                    <div class="prediction-box">
                        <h3>Absenteeism Risk: <span class="risk-indicator">{{ graph_data.risk_level }}</span></h3>
                        <p>Predicted Rate: {{ "%.1f"|format(graph_data.predicted_value) }}%</p>
                        <p>Flu Risk: {{ "%.1f"|format(graph_data.flu_risk) }}%</p>
                        <p>ER Visit Rate: {{ "%.1f"|format(graph_data.er_rate) }}%</p>
                        <p><strong>Weather:</strong> {{ graph_data.weather_desc }}</p>
                        <p><strong>Illness Reasons:</strong> {{ graph_data.illness_reasons }}</p>
                    </div>
                    
                    <pre>{{ output }}</pre>
                {% endif %}
            {% else %}
                <div style="padding-top: 50px;">
                    <h2>Welcome to Karnataka School Absenteeism Predictor</h2>
                    <p>Enter a pincode and date to get predictions</p>
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    output = ""
    graph_data = {
        "location": "Unknown Location",
        "prediction_date": "",
        "risk_level": "Unknown",
        "risk_color": "#cccccc",
        "weather_data": {
            "current": {},
            "averages": {},
            "combined": {}
        },
        "weather_desc": "",
        "illness_reasons": "",
        "predicted_value": 0,
        "flu_risk": 0,
        "er_rate": 0
    }
    
    if request.method == "POST":
        try:
            pincode = request.form["pincode"]
            date = request.form["date"]
            prediction_result, new_graph_data = predict_weather_illness_and_absenteeism(pincode, date)
            output = prediction_result
            graph_data.update(new_graph_data)
        except Exception as e:
            output = f"❌ Error processing request: {str(e)}"
    
    return render_template_string(HTML_PAGE, 
                               output=output,
                               graph_data=graph_data)

@app.route("/plot.png")
def plot_png():
    fig, ax = plt.subplots(figsize=(10, 5))
    
    try:
        values = [float(v) for v in request.args.getlist('values')]
        risk_color = request.args.get('risk_color', '#3498db')
        
        if values:
            ax.bar(range(len(values)), values, color=risk_color, alpha=0.7)
            ax.set_title("Absenteeism Prediction")
            ax.set_ylabel("Percentage (%)")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center')
            
    except Exception as e:
        ax.text(0.5, 0.5, 'Error Generating Graph', ha='center', va='center', color='red')
    
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=120)
    plt.close(fig)
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True, port=5000)