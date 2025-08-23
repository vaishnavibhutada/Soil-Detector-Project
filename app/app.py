import os
import uuid
import sqlite3
import requests
from flask import Flask, render_template, request, g
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json
from flask_babel import Babel, get_locale

app = Flask(__name__)
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATABASE'] = 'soil_results.db'

babel = Babel(app)

# Your OpenWeather API key and endpoints
OPENWEATHER_API_KEY = "6cfa9afa7a0516a1689d67eea871f8d1"
WEATHER_URL = 'https://api.openweathermap.org/data/2.5/weather'
FORECAST_URL = 'https://api.openweathermap.org/data/2.5/forecast'



@app.context_processor
def inject_locale():
    return dict(current_locale=str(get_locale()))

@babel.localeselector
def get_locale():
    lang = request.args.get('lang')
    if lang in ['en', 'hi', 'mr']:
        return lang
    return request.accept_languages.best_match(['en', 'hi', 'mr'])

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(app.config['DATABASE'])
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_filename TEXT,
                predicted_class TEXT,
                confidence REAL,
                latitude REAL,
                longitude REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        db.commit()

model = load_model('../dataset/image_model.h5')

with open('agri_data.json', 'r') as f:
    agri_data = json.load(f)

def prepare_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def save_result_to_db(filename, predicted_class, confidence, lat, lon):
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        INSERT INTO history (image_filename, predicted_class, confidence, latitude, longitude)
        VALUES (?, ?, ?, ?, ?)
    ''', (filename, predicted_class, confidence, lat, lon))
    db.commit()

def get_weather_data(lat, lon):
    if not lat or not lon:
        return None
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(WEATHER_URL, params=params)
        data = response.json()
        if data.get("cod") != 200:
            return None
        return data
    except Exception as e:
        print("Weather API error:", e)
        return None

def generate_watering_advice(weather_data, soil_moisture):
    if not weather_data:
        return "Weather data not available."
    
    rain = weather_data.get('rain', {}).get('1h', 0)
    humidity = weather_data.get('main', {}).get('humidity', 0)
    temp = weather_data.get('main', {}).get('temp', 0)

    if rain > 0:
        return "Rainfall detected recently. No need to water the crops now."
    elif soil_moisture and soil_moisture.lower() == "dry":
        return "Soil is dry and no recent rain. Water the crops adequately."
    elif humidity > 80:
        return "High humidity detected; reduce watering accordingly."
    elif temp > 35:
        return "High temperature detected; increase watering frequency."
    else:
        return "Water as per normal crop requirements."

def get_agri_news():
    url = f"https://newsapi.org/v2/top-headlines?category=science&apiKey={NEWSAPI_API_KEY}&q=agriculture"
    try:
        response = requests.get(url)
        news = response.json()
        if news['status'] == 'ok':
            return news['articles'][:5]
        return []
    except Exception as e:
        print("News API error:", e)
        return []

govt_schemes = [
    {
        "name": "PM-KISAN",
        "details": "Provides income support of ₹6,000 per year to small and marginal farmers.",
        "link": "https://pmkisan.gov.in/"
    },
    {
        "name": "Fertilizer Subsidy",
        "details": "Subsidies on fertilizers to make them affordable for farmers.",
        "link": "https://fertilizer.gov.in/"
    },
    {
        "name": "Minimum Support Price (MSP)",
        "details": "Government fixed price to protect farmers from price fluctuations.",
        "link": "https://agricoop.nic.in/"
    }
]

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        files = request.files.getlist("images")
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')

        if files:
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            
            for file in files:
                if file and file.filename != '':
                    filename = f"{uuid.uuid4().hex}.jpg"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)

                    img_array = prepare_image(filepath)
                    prediction = model.predict(img_array)
                    class_index = np.argmax(prediction)
                    class_labels = ['Alluvial', 'Black', 'Clay', 'Red']
                    soil_type = class_labels[class_index]
                    confidence = prediction[0][class_index] * 100

                    lat = float(latitude) if latitude else None
                    lon = float(longitude) if longitude else None
                    save_result_to_db(filename, soil_type, confidence, lat, lon)

                    suggestions = agri_data.get(soil_type, {})
                    crops = [f"{crop['name']} (Varieties: {', '.join(crop['varieties'])}) - {crop['details']}" for crop in suggestions.get('crops', [])]
                    watering_tips = suggestions.get('watering_tips')
                    fertilizers = [f"{fert['type']}: {fert['recommendation']}" for fert in suggestions.get('fertilizers', [])]
                    soil_health = {
                        'moisture': suggestions.get('soil_health', {}).get('moisture_status'),
                        'nutrient_deficiency': suggestions.get('soil_health', {}).get('nutrient_deficiency')
                    }

                    weather_data = get_weather_data(lat, lon)
                    watering_advice = generate_watering_advice(weather_data, soil_health.get('moisture'))

                    results.append({
                        "soil_type": soil_type,
                        "confidence": round(confidence, 2),
                        "filename": filename,
                        "crops": crops,
                        "watering_tips": watering_tips,
                        "fertilizers": fertilizers,
                        "soil_health": soil_health,
                        "weather": weather_data,
                        "watering_advice": watering_advice
                    })

        return render_template("result.html", results=results)


    return render_template("index.html", get_locale=get_locale)

@app.route('/history')
def history():
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT image_filename, predicted_class, confidence, latitude, longitude, timestamp FROM history ORDER BY timestamp DESC')
    records = cursor.fetchall()
    return render_template('history.html', records=records)

@app.route("/agri-news")
def agri_news():
    api_key = "0bfd660d8c394f7bac5beb088495dfa1"
    url = f"https://newsapi.org/v2/everything?q=agriculture+farming&language=en&pageSize=12&apiKey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        articles = data.get("articles", [])
    except Exception as e:
        print("Error fetching news:", e)
        articles = []
    return render_template("agri_news.html", articles=articles)
@app.route("/govt-schemes")
def govt_schemes_page():
    schemes = [
        {
            "name": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
            "description": "Provides crop insurance to farmers against natural calamities.",
            "benefits": "Risk coverage for crops from non-preventable natural calamities.",
            "eligibility": "All farmers including sharecroppers and tenant farmers.",
            "link": "https://pmfby.gov.in/"
        },
        {
            "name": "Paramparagat Krishi Vikas Yojana (PKVY)",
            "description": "Promotes organic farming through cluster-based approach.",
            "benefits": "Financial assistance up to ₹50,000 per hectare for 3 years.",
            "eligibility": "Farmers practicing organic farming in clusters.",
            "link": "https://pgsindia-ncof.gov.in/PKVY/index.aspx"
        },
        {
            "name": "PM-Kisan Samman Nidhi",
            "description": "Provides income support to small and marginal farmers.",
            "benefits": "₹6,000 per year in three equal installments.",
            "eligibility": "All landholding farmers' families.",
            "link": "https://pmkisan.gov.in/"
        },
        {
            "name": "National Mission on Natural Farming (NMNF)",
            "description": "Promotes natural farming practices across the country.",
            "benefits": "Support for training, certification, and marketing of natural farming products.",
            "eligibility": "Farmers adopting natural farming methods.",
            "link": "https://agricoop.nic.in/"
        },
        {
            "name": "Kisan Credit Card (KCC) Scheme",
            "description": "Provides timely credit support to farmers for their cultivation needs.",
            "benefits": "Short-term credit for crop cultivation and other needs.",
            "eligibility": "All farmers engaged in agriculture and allied activities.",
            "link": "https://www.nabard.org/"
        }
    ]
    return render_template("govt_schemes.html", schemes=schemes)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
