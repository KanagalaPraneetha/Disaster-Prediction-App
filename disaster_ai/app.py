# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
from geopy.geocoders import Nominatim

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("models/disaster_model.pkl", "rb"))

# OpenWeatherMap API configuration
API_KEY = "9bed7bfeae6e8e77c532681df90d03a3"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get location input
    location = request.form["location"]

    # Fetch weather data from OpenWeatherMap API
    try:
        response = requests.get(BASE_URL, params={"q": location, "appid": API_KEY, "units": "metric"})
        data = response.json()

        if response.status_code == 200:
            temperature = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]
            lat = data["coord"]["lat"]  # Latitude
            lon = data["coord"]["lon"]  # Longitude

            # Predict using the model
            input_data = np.array([[temperature, humidity, wind_speed]])
            prediction = model.predict(input_data)

            # Interpret prediction
            result = "Disaster Likely" if prediction[0] == 1 else "No Disaster"
            return render_template("index.html", location=location, result=result, weather=data, lat=lat, lon=lon)
        else:
            error_message = data.get("message", "Error fetching weather data")
            return render_template("index.html", error=error_message)
    except Exception as e:
        return render_template("index.html", error=str(e))

@app.route("/map/<location>")
def show_map(location):
    # Geolocate the location using geopy
    geolocator = Nominatim(user_agent="disaster-ai")
    location_data = geolocator.geocode(location)

    # Handle case where location is not found
    if not location_data:
        return render_template("result.html", result="Location not found")

    # Pass the coordinates (latitude and longitude) to the template
    return render_template(
        "map.html", 
        latitude=location_data.latitude, 
        longitude=location_data.longitude, 
        location=location  # Optional: Pass the location name to display on the map
    )

if __name__ == "__main__":
    app.run(debug=True, threaded=False)