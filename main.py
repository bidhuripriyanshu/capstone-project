import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Base and Data Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. API: Fetch realtime weather data from API and update CSV
# -------------------------------------------------------------
def fetch_api_data_and_update_csv(latitude=23.077080, longitude=76.85131):
    """Fetch hourly weather metrics from Open-Meteo API and update data/realtime.csv"""
    print("\n[Step 1] Fetching realtime weather data from API...")
    today = datetime.now()
    start_date = today.strftime('%Y-%m-%d')
    end_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}&"
        f"hourly=temperature_2m,windspeed_10m,shortwave_radiation,cloudcover,relative_humidity_2m&"
        f"start_date={start_date}&end_date={end_date}&timezone=auto"
    )
    
    realtime_file = os.path.join(DATA_DIR, "realtime.csv")
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()['hourly']
            df_api = pd.DataFrame({
                'Date/Time': data['time'],
                'Ambient_Temperature': data['temperature_2m'],
                'Wind_Speed': data['windspeed_10m'],
                'Solar_Irradiance': data['shortwave_radiation'],
                'Cloud_Cover': data['cloudcover'],
                'Humidity': data['relative_humidity_2m']
            })
            df_api.to_csv(realtime_file, index=False)
            print(f"  [OK] API data updated and saved to '{realtime_file}' ({len(df_api)} rows)")
            return df_api
        else:
            print(f"  [!] API returned status code {response.status_code}. Using local CSV fallback.")
    except Exception as e:
        print(f"  [!] API request exception: {e}. Using local CSV fallback.")
        
    if os.path.exists(realtime_file):
        return pd.read_csv(realtime_file)
    return None

# -------------------------------------------------------------
# 2. Clean CSV: Preprocess and format features
# -------------------------------------------------------------
def clean_csv(df):
    """Clean missing values, correct headers, and create time features"""
    df = df.copy()
    
    # Correct renamed/truncated columns (e.g. 'Hu' -> 'Humidity')
    if 'Hu' in df.columns and 'Humidity' not in df.columns:
        df.rename(columns={'Hu': 'Humidity'}, inplace=True)
        
    # Standardize Date/Time & Hour
    if 'Date/Time' in df.columns:
        df['Date/Time'] = pd.to_datetime(df['Date/Time'])
        if 'Hour' not in df.columns:
            df['Hour'] = df['Date/Time'].dt.hour
    elif 'Hour' not in df.columns:
        df['Hour'] = 12

    # Cyclical hour features
    df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    
    # Impute missing numerical values
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # Ensure target variables exist for model training
    if 'solar_energy_kwh' not in df.columns:
        if 'Solar_Irradiance' in df.columns and df['Solar_Irradiance'].max() > 0:
            df['solar_energy_kwh'] = df['Solar_Irradiance'] * 0.015 + df['Ambient_Temperature'] * 0.02
        else:
            df['solar_energy_kwh'] = ((100 - df['Cloud_Cover']) * 0.05 + df['Ambient_Temperature'] * 0.03)
            df['solar_energy_kwh'] = df['solar_energy_kwh'] * (df['Hour'].between(6, 18).astype(int) * 0.9 + 0.1)
        df['solar_energy_kwh'] = df['solar_energy_kwh'].clip(lower=0)

    if 'wind_energy_kwh' not in df.columns:
        df['wind_energy_kwh'] = (df['Wind_Speed'] ** 3) * 0.01
        df['wind_energy_kwh'] = df['wind_energy_kwh'].clip(lower=0)
        
    return df

# -------------------------------------------------------------
# 3. Model Training
# -------------------------------------------------------------
def train_models(df_clean):
    """Train Solar and Wind Energy Prediction Models"""
    print("\n[Step 3] Model Training...")
    
    feature_candidates = [
        'Ambient_Temperature', 'Wind_Speed', 'Cloud_Cover', 'Humidity',
        'Solar_Irradiance', 'hour_sin', 'hour_cos'
    ]
    features = [f for f in feature_candidates if f in df_clean.columns]
    print(f"  Features used for training: {features}")
    
    X = df_clean[features]
    y_solar = df_clean['solar_energy_kwh']
    y_wind = df_clean['wind_energy_kwh']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Solar Energy Model
    solar_model = RandomForestRegressor(n_estimators=100, random_state=42)
    solar_model.fit(X_scaled, y_solar)
    
    # Train Wind Energy Model
    wind_model = RandomForestRegressor(n_estimators=100, random_state=42)
    wind_model.fit(X_scaled, y_wind)
    
    print("  [OK] Solar & Wind RandomForest models trained successfully.")
    return solar_model, wind_model, scaler, features

# -------------------------------------------------------------
# 4 & 5. Working of Model on Clean CSV & Hourly Predictions Table
# -------------------------------------------------------------
def predict_hourly_energy(solar_model, wind_model, scaler, features, df_realtime_clean):
    """Predict hourly solar and wind energy output on clean realtime weather CSV"""
    print("\n[Step 4] Running model predictions on clean realtime weather data...")
    
    for f in features:
        if f not in df_realtime_clean.columns:
            df_realtime_clean[f] = 0
            
    X_realtime = df_realtime_clean[features]
    X_realtime_scaled = scaler.transform(X_realtime)
    
    solar_preds = np.maximum(0, np.round(solar_model.predict(X_realtime_scaled), 2))
    wind_preds = np.maximum(0, np.round(wind_model.predict(X_realtime_scaled), 2))
    
    # Construct Output Table matching Phase 1 diagram
    result_df = pd.DataFrame({
        'DateTime': df_realtime_clean['Date/Time'].dt.strftime('%Y-%m-%d %H:%M'),
        'Predicted Solar (kW/h)': solar_preds,
        'Predicted Wind (kW/h)': wind_preds
    })
    return result_df

def main():
    print("=" * 70)
    print(" Weather-based Energy Prediction System Realtime Data (Phase 1)")
    print("=" * 70)
    
    # 1. Fetch data from API and update CSV
    df_api = fetch_api_data_and_update_csv()
    
    # Load training dataset (merged.csv or weather_last_year_data.csv)
    merged_path = os.path.join(DATA_DIR, "merged.csv")
    historical_path = os.path.join(DATA_DIR, "weather_last_year_data .csv")
    
    if os.path.exists(merged_path):
        train_raw = pd.read_csv(merged_path)
    elif os.path.exists(historical_path):
        train_raw = pd.read_csv(historical_path)
    else:
        train_raw = df_api
        
    # 2. Clean CSV
    print("\n[Step 2] Cleaning raw dataset...")
    train_clean = clean_csv(train_raw)
    realtime_clean = clean_csv(df_api)
    print("  [OK] Data cleaning completed.")
    
    # 3. Model Training
    solar_model, wind_model, scaler, features = train_models(train_clean)
    
    # 4 & 5. Working of Model & Output Hourly Predictions Table
    predictions_df = predict_hourly_energy(solar_model, wind_model, scaler, features, realtime_clean)
    
    print("\n" + "=" * 70)
    print(" [Output] Hourly bases per energy predict hogi")
    print("=" * 70)
    print(predictions_df.to_string(index=False))
    print("=" * 70)
    
    # Save output JSON forecasts
    forecast_date = datetime.now().strftime('%Y-%m-%d')
    solar_output = {
        "date": forecast_date,
        "granularity": "hourly",
        "forecast_series_kwh": list(predictions_df['Predicted Solar (kW/h)']),
        "total_generation_kwh": round(float(predictions_df['Predicted Solar (kW/h)'].sum()), 2)
    }
    wind_output = {
        "date": forecast_date,
        "granularity": "hourly",
        "forecast_series_kwh": list(predictions_df['Predicted Wind (kW/h)']),
        "total_generation_kwh": round(float(predictions_df['Predicted Wind (kW/h)'].sum()), 2)
    }
    
    with open(os.path.join(BASE_DIR, 'solar_energy_forecast.json'), 'w') as f:
        json.dump(solar_output, f, indent=2)
    with open(os.path.join(BASE_DIR, 'wind_energy_forecast.json'), 'w') as f:
        json.dump(wind_output, f, indent=2)
    
    print(f"\n[OK] Forecast JSON files saved to '{BASE_DIR}'")

if __name__ == "__main__":
    main()