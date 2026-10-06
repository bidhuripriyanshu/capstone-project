import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# Set your project location (example: Delhi)
latitude = 23.077080
longitude = 76.85131

# Calculate tomorrow's date
start_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
end_date = start_date

# Open-Meteo API URL for weather
url = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={latitude}&longitude={longitude}&"
    f"hourly=temperature_2m,windspeed_10m,shortwave_radiation,cloudcover,relative_humidity_2m&"
    f"start_date={start_date}&end_date={end_date}&timezone=auto"
    
)

response = requests.get(url)
if response.status_code == 200:
    data = response.json()['hourly']
    df = pd.DataFrame({
        'Date/Time': data['time'],
        'Ambient_Temperature': data['temperature_2m'],
        'Wind_Speed': data['windspeed_10m'],
        'Cloud_Cover': data['cloudcover'],
        'Humidity': data['relative_humidity_2m']
    })
    
    # Save inside Data folder
    folder_path = "Data"
    os.makedirs(folder_path, exist_ok=True)   # Ensure folder exists
    file_name = os.path.join(folder_path, "realtime.csv")
    
    df.to_csv(file_name, index=False)
    print(f"Weather forecast saved to {file_name}")
else:
    print("Failed to fetch weather data")
