# ✈️ Air Passenger SARIMAX Forecasting

A time-series forecasting project that analyzes historical monthly international airline passenger traffic and predicts future passenger demand using the **SARIMAX (Seasonal ARIMA with Exogenous Variables)** model.

The project includes data preprocessing, exploratory data analysis, stationarity testing, differencing, ACF/PACF analysis, SARIMAX forecasting, model evaluation, airline-wise forecasting, and an interactive **Streamlit dashboard**.

---

## 📌 Project Overview

Airline passenger demand changes over time and often contains trends and seasonal patterns.

This project uses historical passenger data to:

* Analyze monthly passenger traffic
* Identify trends and seasonal patterns
* Test whether the time series is stationary
* Apply differencing when required
* Build a SARIMAX forecasting model
* Forecast passenger traffic for the next 12 months
* Evaluate forecasting performance
* Generate airline-wise forecasts
* Present results through an interactive Streamlit dashboard

### 🎯 Main Objective

> To predict future international airline passenger traffic using historical time-series data and SARIMAX forecasting.

---

## 🧠 Technologies Used

| Technology   | Purpose                    |
| ------------ | -------------------------- |
| Python       | Programming language       |
| Pandas       | Data processing            |
| NumPy        | Numerical operations       |
| Matplotlib   | Data visualization         |
| Statsmodels  | ADF test and SARIMAX       |
| Plotly       | Interactive visualizations |
| Streamlit    | Web dashboard              |
| Scikit-learn | Model evaluation           |
| Git/GitHub   | Version control            |

---

## 📊 Airlines Included

The dataset contains monthly passenger information for:

* Emirates
* Qatar Airways
* Etihad Airways
* Singapore Airlines
* SriLankan Airlines
* British Airways
* Lufthansa
* Malaysia Airlines
* Thai Airways
* Air Arabia
* Oman Air
* Saudia

---

# 🔄 Project Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Time-Series Visualization
   ↓
ADF Stationarity Test
   ↓
First Differencing
   ↓
Seasonal Differencing
   ↓
ACF & PACF Analysis
   ↓
SARIMAX Model
   ↓
12-Month Forecast
   ↓
Model Evaluation
   ↓
Airline-wise Forecast
   ↓
Streamlit Dashboard
```

---

# 🛠️ Project Steps

## 1. Data Collection

Historical monthly passenger data is stored in:

```text
passengers.csv
```

The dataset contains passenger counts for multiple international airlines.

---

## 2. Data Preprocessing

The `Month` column is converted into a datetime format and the dataset is sorted chronologically.

```python
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month")
```

The passenger values of all airlines are then combined to calculate monthly total passenger traffic.

---

## 3. Exploratory Data Analysis

EDA is performed to understand:

* Dataset structure
* Passenger distribution
* Monthly passenger traffic
* Missing values
* Statistical characteristics
* Airline-wise passenger patterns

---

## 4. Time-Series Visualization

The project generates:

```text
total_passenger_trend.png
```

and

```text
airline_wise_trend.png
```

These visualizations help identify long-term trends and seasonal variations.

---

## 5. Stationarity Testing

The **Augmented Dickey-Fuller (ADF) test** is used to determine whether the passenger time series is stationary.

The hypothesis is:

$$
H_0 = \text{Series is non-stationary}
$$

$$
H_1 = \text{Series is stationary}
$$

A p-value below 0.05 indicates rejection of the null hypothesis.

---

## 6. Differencing

When the original series is non-stationary, first-order differencing is applied:

$$
Y'_t = Y_t - Y_{t-1}
$$

This helps remove the trend component.

---

## 7. Seasonal Differencing

Since the dataset contains monthly observations, a seasonal period of:

$$
s = 12
$$

is used.

Seasonal differencing helps remove yearly seasonal patterns.

---

## 8. ACF & PACF Analysis

Autocorrelation and partial autocorrelation values are analyzed to help determine appropriate AR and MA parameters for the SARIMAX model.

---

## 9. SARIMAX Model

The processed time series is used to train the SARIMAX forecasting model.

A SARIMAX model can represent:

$$
SARIMA(p,d,q)(P,D,Q)_s
$$

where:

* \(p\) = AR order
* \(d\) = differencing order
* \(q\) = MA order
* \(P\) = seasonal AR order
* \(D\) = seasonal differencing order
* \(Q\) = seasonal MA order
* \(s\) = seasonal period

For monthly data:

$$
s=12
$$

---

# 🔮 10. Forecasting

The trained model predicts future passenger traffic.

The current implementation produces a:

### **12-month forecast**

The forecast is saved as:

```text
passenger_forecast.csv
```

The forecast contains future months and predicted passenger values.

---

# 📈 11. Model Evaluation

The forecasting model is evaluated using:

### MAE

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
$$

### RMSE

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
$$

### MAPE

$$
MAPE =
\frac{100}{n}
\sum_{i=1}^{n}
\left|
\frac{y_i-\hat{y}_i}{y_i}
\right|
$$

The recorded model evaluation was:

| Metric |     Result |
| ------ | ---------: |
| MAE    | 120,728.96 |
| RMSE   | 126,181.70 |
| MAPE   |      3.93% |

These values are specific to the dataset and evaluation split used in this project.

---

# ✈️ 12. Airline-wise Forecast

The project also provides forecasting analysis for individual airlines.

This allows the user to examine future passenger demand for airlines such as:

* Emirates
* Qatar Airways
* Etihad Airways
* Singapore Airlines
* British Airways
* Lufthansa
* Oman Air
* Saudia

---

# 🌐 13. Streamlit Dashboard

The project includes an interactive Streamlit application.

The dashboard provides:

* 📊 Passenger statistics
* ✈️ Airline selection
* 📈 Historical passenger analysis
* 🔮 Future passenger forecasting
* 📋 Forecast tables
* 📉 Model evaluation
* 📊 Interactive Plotly visualizations

The interface is designed to work in both desktop browsers and mobile browsers.

---

# 📁 Project Structure

```text
Air_Passenger_SARIMAX/
│
├── venv/
│
├── passengers.csv
│
├── main.py
├── app.py
│
├── passenger_forecast.csv
├── airline_wise_forecast.csv
├── model_evaluation.csv
├── acf_pacf_values.csv
│
├── total_passenger_trend.png
├── airline_wise_trend.png
│
├── model.pkl
│
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Go to the project directory:

```bash
cd Air_Passenger_SARIMAX
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```powershell
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn plotly streamlit
```

---

# ▶️ Running the Project

## Run the forecasting program

```bash
python main.py
```

## Run the Streamlit dashboard

```bash
streamlit run app.py
```

The Streamlit application will open in the browser.

---

# 📱 Dashboard

The Streamlit dashboard provides an easy-to-use interface for analyzing passenger traffic and viewing future forecasts.

Users can select an airline, inspect historical passenger data, view forecast values, and evaluate model performance.

---

# 💡 Applications

This forecasting system can support:

* Airline demand planning
* Flight scheduling
* Capacity planning
* Airport resource planning
* Staff planning
* Passenger demand analysis
* Business forecasting
* Seasonal demand analysis

---

# ⚠️ Limitations

* Forecast accuracy depends on the quality and size of the historical dataset.
* SARIMAX performance can change when new data is added.
* External factors such as fuel prices, economic conditions, pandemics, geopolitical events, and sudden travel restrictions are not necessarily represented in the historical dataset.
* Forecasts should therefore be treated as estimates rather than guaranteed future passenger counts.

---

# 🚀 Future Enhancements

Possible future improvements include:

* Real-time airline data integration
* Weather and economic variables
* Automatic SARIMAX parameter optimization
* Comparison with Prophet, LSTM, XGBoost and other models
* Real-time forecast updates
* Cloud deployment
* User authentication
* Database integration
* Automated model retraining
* Airline-specific forecasting models

---

# 🎓 Academic Information

### Project Title

**Air Passenger Traffic Forecasting Using SARIMAX**

### Domain

**Time-Series Forecasting / Machine Learning**

### Main Algorithm

**SARIMAX**

### Frontend / Dashboard

**Streamlit**

### Programming Language

**Python**

---

# 👨‍💻 Author

**Mohammed Rishan P**

B.Tech Computer Science & Engineering
Artificial Intelligence & Machine Learning

---

## ⭐ Project Summary

```text
Historical Airline Data
        ↓
Data Preprocessing
        ↓
Stationarity Analysis
        ↓
Differencing
        ↓
Seasonality Analysis
        ↓
SARIMAX
        ↓
12-Month Passenger Forecast
        ↓
Model Evaluation
        ↓
Streamlit Dashboard
```

> **Predicting tomorrow's passenger demand using yesterday's data.** ✈️📊
