# 🌦️ Weather Forecast ML

A machine learning app that predicts **tomorrow's maximum temperature** for a
selected city, using 6 years of real weather history. It also shows how
accurate the model is compared with a simple baseline.

**Live app:** <your-streamlit-link>

---

## Problem
Can a machine learning model predict tomorrow's temperature using only the
last three days of weather?

## Solution
A Random Forest Regressor learns the pattern between recent weather
(today, yesterday, two days ago) and the next day's temperature. The model is
trained inside the app from live historical data, so no data files are needed.

## Features
- Choose from Chennai, Bengaluru, Delhi, or Mumbai
- Automatic data download from the free Open-Meteo API (no API key)
- Model training and evaluation inside the app
- Comparison with a baseline guess ("tomorrow = today")
- Live prediction for tomorrow, with the input data shown

## Tech Stack
| Tool | Use |
|---|---|
| Python | Main language |
| Pandas | Data handling and feature creation |
| Scikit-learn | Random Forest model and error metric |
| Requests | Calling the weather API |
| Streamlit | Web interface |

## Results
| Measure | Value |
|---|---|
| Model error (MAE) | <your value> °C |
| Baseline error (tomorrow = today) | <your value> °C |
| Improvement over baseline | <your value> % |

> Run the app, read the two numbers shown on the page, and fill in this table.
> A smaller error is better.

## How It Works
```
Open-Meteo API (2020 → last week)
   │
   ▼
Create features: today + yesterday + 2 days ago
   │
   ▼
Create target: tomorrow's max temperature
   │
   ▼
Split by time: first 80% train, last 20% test
   │
   ▼
Train Random Forest → measure error → compare with baseline
   │
   ▼
Fetch the latest 3 days → predict tomorrow
```

## Code Explanation
| Part | What it does |
|---|---|
| `add_features()` | Uses `shift()` to add yesterday's and the day-before's values as extra columns (called **lag features**). |
| `tomorrow_temp` column | The answer the model must learn, which is the next day's temperature. |
| Time-based split | Trains on older days and tests on newer days. This copies real use and avoids data leakage. |
| `RandomForestRegressor` | Many decision trees that vote together to give one prediction. |
| `mean_absolute_error` | The average mistake in °C. |
| Baseline | Predicts "tomorrow = today". A useful model must beat this. |
| `@st.cache_resource` | Trains once per city and reuses the model, so the app stays fast. |

## Run Locally
```bash
git clone https://github.com/<your-username>/14-weather-forecast-ml.git
cd 14-weather-forecast-ml
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```
No API key is needed. The first load takes about a minute while the model trains.

## Limitations
- Uses only recent temperature, rain, and min temperature, with no humidity,
  pressure, or cloud data
- Weather is hard to predict beyond a short range, so this is a learning
  project and not a replacement for professional forecasts

## Future Improvements
- Add humidity, pressure, and wind features
- Compare Linear Regression, XGBoost, and Random Forest in a table
- Show a feature importance chart
- Add a rain prediction using a classifier
- Wrap the model in a FastAPI endpoint

## What I Learned
Time-series feature engineering, avoiding data leakage with a time-based split,
evaluating a model against a baseline, working with REST APIs, and deploying
an ML app.