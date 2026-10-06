# Weather-Based Energy Prediction System (Phase 1)

A Machine Learning system that fetches real-time weather data via API, cleans dataset inputs, trains Random Forest regression models, and outputs 24/48-hour hourly solar and wind energy generation forecasts.

---

## 🏗️ System Architecture Flowchart (Phase 1 Epics)

The system strictly follows the Phase 1 Epics workflow architecture:

```mermaid
flowchart TD
    subgraph API ["API (Realtime Weather Data)"]
        A[Open-Meteo API] --> B[Weather Parameters:<br/>• Temperature °C<br/>• Wind Speed m/s<br/>• Solar Irradiance W/m²<br/>• Humidity %<br/>• Cloud Cover %]
    end

    subgraph ML ["Machine Learning Models"]
        B --> C[Fetch data from API and update CSV]
        C --> D[Clean CSV]
        D --> E[Model Training<br/>RandomForestRegressor]
        E --> F[Working of model on the bases of clean CSV]
    end

    subgraph Output ["Hourly Predictions Output"]
        F --> G[Hourly bases per energy predict hogi]
        G --> H[Output Table:<br/>DateTime | Predicted Solar kW/h | Predicted Wind kW/h]
        G --> I[JSON Forecast Outputs:<br/>solar_energy_forecast.json<br/>wind_energy_forecast.json]
    end
```

### Flow Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                      Phase 1 Epics Architecture                         │
└─────────────────────────────────────────────────────────────────────────┘
  [ API ] ──► Weather Data (Temp, Wind Speed, Solar Rad, Cloud, Humidity)
    │
    ▼
  [ Machine Learning Models ]
    ├──► 1. Fetch data from API and update CSV (data/realtime.csv)
    ├──► 2. Clean CSV (data/merged.csv & data/realtime.csv)
    ├──► 3. Model Training (Solar & Wind RandomForest Models)
    └──► 4. Working of model on clean CSV
    │
    ▼
  [ Output: Hourly bases per energy predict hogi ]
    └──► Table: DateTime | Predicted Solar (kW/h) | Predicted Wind (kW/h)
```

---

## 📁 Repository Structure

```text
Weather-Based-Energy-Prediction-System-Realtime-Data/
├── data/
│   ├── weather_last_year_data .csv   # Historical training dataset
│   ├── realtime.csv                   # Updated realtime weather API data
│   ├── merged.csv                     # Merged & cleaned historical/realtime data
│   └── inputs_defaults.json           # Default parameter configuration
├── jupyter-notebook/
│   └── model_precessing.ipynb         # Data analysis & ML model exploration notebook
├── main.py                            # Main Phase 1 pipeline execution script
├── weather_forecast.py                # Standalone API fetch script
├── solar_energy_forecast.json         # Generated Solar energy forecast output
├── wind_energy_forecast.json          # Generated Wind energy forecast output
└── README.md                          # Phase 1 Documentation
```

---

## 🚀 Quick Start & Execution

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn requests
```

### 2. Run the Main Pipeline
Run the Phase 1 prediction pipeline from the project directory:

```bash
python main.py
```

**What `main.py` executes:**
1. **Fetch API Data**: Retrieves realtime weather metrics from Open-Meteo API and updates `data/realtime.csv`.
2. **Clean CSV**: Cleans raw datasets, imputes missing values, standardizes column headers, and constructs cyclical time features (`hour_sin`, `hour_cos`).
3. **Model Training**: Trains separate `RandomForestRegressor` models for Solar and Wind energy predictions.
4. **Working of Model**: Applies trained models on the cleaned realtime weather forecast data.
5. **Output Predictions**: Prints the hourly prediction table (`DateTime`, `Predicted Solar (kW/h)`, `Predicted Wind (kW/h)`) and saves `solar_energy_forecast.json` & `wind_energy_forecast.json`.

---

## 📊 Sample Output Table

```text
======================================================================
 [Output] Hourly bases per energy predict hogi
======================================================================
        DateTime  Predicted Solar (kW/h)  Predicted Wind (kW/h)
2026-10-06 00:00                    1.70                   7.16
2026-10-06 01:00                    1.08                   3.17
2026-10-06 02:00                    0.88                   1.95
2026-10-06 03:00                    1.21                   1.31
2026-10-06 04:00                    1.36                   0.93
...
```

---

## ⚙️ Features & Weather Input Parameters

| Parameter | Unit | Description |
|---|---|---|
| **Ambient_Temperature** | °C | Air temperature |
| **Wind_Speed** | m/s | Wind speed at 10m height |
| **Solar_Irradiance** | W/m² | Shortwave solar radiation intensity |
| **Humidity** | % | Relative humidity |
| **Cloud_Cover** | % | Cloud coverage percentage |
| **hour_sin / hour_cos** | Numeric | Cyclical hourly trigonometric features |