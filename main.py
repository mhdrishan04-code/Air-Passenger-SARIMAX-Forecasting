import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# STEP 1 - DATA COLLECTION
# =========================================================

print("\n==========================================")
print("STEP 1 - DATA COLLECTION")
print("==========================================")

df = pd.read_csv("passengers.csv")

print("Dataset loaded successfully.")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

print("\nSTEP 1 COMPLETED SUCCESSFULLY")


# =========================================================
# STEP 2 - DATA PREPROCESSING
# =========================================================

print("\n==========================================")
print("STEP 2 - DATA PREPROCESSING")
print("==========================================")

df["Month"] = pd.to_datetime(df["Month"])

df = df.sort_values("Month")

df = df.drop_duplicates(subset="Month")

print("Date conversion completed.")
print("Data sorted by Month.")

print("\nDate Range:")
print(
    df["Month"].min(),
    "to",
    df["Month"].max()
)

print("\nSTEP 2 COMPLETED SUCCESSFULLY")


# =========================================================
# AIRLINE COLUMNS
# =========================================================

airlines = [
    "Emirates",
    "Qatar Airways",
    "Etihad Airways",
    "Singapore Airlines",
    "SriLankan Airlines",
    "British Airways",
    "Lufthansa",
    "Malaysia Airlines",
    "Thai Airways",
    "Air Arabia",
    "Oman Air",
    "Saudia"
]


# Check airline columns
missing_columns = [
    airline
    for airline in airlines
    if airline not in df.columns
]

if missing_columns:

    print("\nMissing airline columns:")
    print(missing_columns)

    raise ValueError(
        "Required airline columns are missing from passengers.csv"
    )


# =========================================================
# TOTAL PASSENGERS
# =========================================================

df["Total_Passengers"] = (
    df[airlines].sum(axis=1)
)

print("\n--- MONTHLY TOTAL PASSENGERS ---")

print(
    df[
        ["Month", "Total_Passengers"]
    ].head(10)
)


# =========================================================
# STEP 3 - EDA
# =========================================================

print("\n==========================================")
print("STEP 3 - EDA")
print("==========================================")

print("\nDataset Information:")

df.info()

print("\nMissing Values:")

print(
    df.isnull().sum()
)

print("\nStatistical Summary:")

print(
    df["Total_Passengers"].describe()
)

print("\nTotal Passenger Statistics:")

print(
    "Maximum :",
    df["Total_Passengers"].max()
)

print(
    "Minimum :",
    df["Total_Passengers"].min()
)

print(
    "Average :",
    df["Total_Passengers"].mean()
)

print("\nSTEP 3 COMPLETED SUCCESSFULLY")


# =========================================================
# STEP 4 - TIME SERIES VISUALIZATION
# =========================================================

print("\n==========================================")
print("STEP 4 - TIME SERIES VISUALIZATION")
print("==========================================")


# ---------------------------------------------------------
# Graph 1 - Total Passenger Trend
# ---------------------------------------------------------

plt.figure(figsize=(14, 7))

plt.plot(
    df["Month"],
    df["Total_Passengers"],
    marker="o",
    linewidth=2
)

plt.title(
    "Monthly International Airline Passenger Traffic"
)

plt.xlabel("Month")

plt.ylabel("Total Passengers")

plt.grid(True)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "total_passenger_trend.png",
    dpi=300
)

plt.close()

print(
    "Total passenger graph saved:"
)

print(
    "total_passenger_trend.png"
)


# ---------------------------------------------------------
# Graph 2 - Airline-wise Trend
# ---------------------------------------------------------

plt.figure(figsize=(15, 8))

for airline in airlines:

    plt.plot(
        df["Month"],
        df[airline],
        label=airline
    )

plt.title(
    "Airline-wise Monthly Passenger Traffic"
)

plt.xlabel("Month")

plt.ylabel("Passengers")

plt.legend()

plt.grid(True)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "airline_wise_trend.png",
    dpi=300
)

plt.close()

print(
    "Airline-wise graph saved:"
)

print(
    "airline_wise_trend.png"
)

print(
    "\nSTEP 4 COMPLETED SUCCESSFULLY"
)


# =========================================================
# STEP 5 - ADF STATIONARITY TEST
# =========================================================

print("\n==========================================")
print("STEP 5 - ADF STATIONARITY TEST")
print("==========================================")

original_series = (
    df["Total_Passengers"]
    .dropna()
)

result = adfuller(
    original_series
)

adf_statistic = result[0]

p_value = result[1]

print(
    "ADF Statistic :",
    adf_statistic
)

print(
    "P-Value       :",
    p_value
)

if p_value <= 0.05:

    print(
        "Result        : STATIONARY"
    )

else:

    print(
        "Result        : NOT STATIONARY"
    )

print(
    "\nSTEP 5 COMPLETED SUCCESSFULLY"
)


# =========================================================
# STEP 6 - FIRST DIFFERENCING
# =========================================================

print("\n==========================================")
print("STEP 6 - FIRST DIFFERENCING")
print("==========================================")

df["Differenced_Passengers"] = (
    df["Total_Passengers"].diff()
)

differenced_series = (
    df["Differenced_Passengers"]
    .dropna()
)

result_diff = adfuller(
    differenced_series
)

adf_statistic_diff = result_diff[0]

p_value_diff = result_diff[1]

print(
    "ADF Statistic :",
    adf_statistic_diff
)

print(
    "P-Value       :",
    p_value_diff
)

if p_value_diff <= 0.05:

    print(
        "Result        : "
        "STATIONARY AFTER DIFFERENCING"
    )

else:

    print(
        "Result        : "
        "STILL NOT STATIONARY"
    )

print(
    "\nSTEP 6 COMPLETED SUCCESSFULLY"
)


# =========================================================
# STEP 7 - SEASONAL DIFFERENCING
# =========================================================

print("\n==========================================")
print("STEP 7 - SEASONAL DIFFERENCING")
print("==========================================")

seasonal_period = 12

seasonally_differenced = (
    differenced_series
    .diff(seasonal_period)
    .dropna()
)

result_seasonal = adfuller(
    seasonally_differenced
)

adf_statistic_seasonal = (
    result_seasonal[0]
)

p_value_seasonal = (
    result_seasonal[1]
)

print(
    "ADF Statistic :",
    adf_statistic_seasonal
)

print(
    "P-Value       :",
    p_value_seasonal
)

if p_value_seasonal <= 0.05:

    print(
        "Result        : "
        "STATIONARY AFTER SEASONAL DIFFERENCING"
    )

else:

    print(
        "Result        : "
        "STILL NOT STATIONARY"
    )

print(
    "\nSTEP 7 COMPLETED SUCCESSFULLY"
)


# =========================================================
# STEP 8 - ACF AND PACF ANALYSIS
# =========================================================

print("\n==========================================")
print("STEP 8 - ACF AND PACF ANALYSIS")
print("==========================================")

acf_values = acf(
    seasonally_differenced,
    nlags=23,
    fft=True
)

print("\nACF Values:")

for lag, value in enumerate(acf_values):

    print(
        f"Lag {lag:2d} : {value:.6f}"
    )


pacf_values = pacf(
    seasonally_differenced,
    nlags=23,
    method="ywm"
)

print("\nPACF Values:")

for lag, value in enumerate(pacf_values):

    print(
        f"Lag {lag:2d} : {value:.6f}"
    )


acf_pacf_df = pd.DataFrame({

    "Lag": range(24),

    "ACF": acf_values,

    "PACF": pacf_values

})

acf_pacf_df.to_csv(
    "acf_pacf_values.csv",
    index=False
)

print(
    "\nACF/PACF values saved as:"
)

print(
    "acf_pacf_values.csv"
)

print(
    "\nSTEP 8 COMPLETED SUCCESSFULLY"
)


# =========================================================
# STEP 9 - SARIMAX MODEL
# =========================================================

print("\n==========================================")
print("STEP 9 - SARIMAX MODEL")
print("==========================================")

ts = (
    df.set_index("Month")
    ["Total_Passengers"]
    .asfreq("MS")
)

ts = ts.interpolate()

test_size = 12

train_data = ts.iloc[:-test_size]

test_data = ts.iloc[-test_size:]

print("\nTraining Data:")

print(
    train_data.index.min(),
    "to",
    train_data.index.max()
)

print("\nTesting Data:")

print(
    test_data.index.min(),
    "to",
    test_data.index.max()
)


# SARIMAX model
model = SARIMAX(

    train_data,

    order=(1, 1, 1),

    seasonal_order=(1, 1, 1, 12),

    enforce_stationarity=False,

    enforce_invertibility=False
)


model_fit = model.fit(
    disp=False
)

print(
    "\nSARIMAX Model Fitted Successfully."
)

print(
    "\nSTEP 9 COMPLETED SUCCESSFULLY"
)


# =========================================================
# STEP 10 - FORECASTING
# =========================================================

print("\n==========================================")
print("STEP 10 - FORECASTING")
print("==========================================")


# ---------------------------------------------------------
# Test Forecast
# ---------------------------------------------------------

test_forecast = (
    model_fit.get_forecast(
        steps=len(test_data)
    )
)

test_forecast_values = (
    test_forecast.predicted_mean
)

test_forecast_values.index = (
    test_data.index
)


# ---------------------------------------------------------
# Final Model Using Complete Dataset
# ---------------------------------------------------------

final_model = SARIMAX(

    ts,

    order=(1, 1, 1),

    seasonal_order=(1, 1, 1, 12),

    enforce_stationarity=False,

    enforce_invertibility=False
)

final_model_fit = (
    final_model.fit(
        disp=False
    )
)


# ---------------------------------------------------------
# Future Forecast - 12 Months
# ---------------------------------------------------------

forecast_steps = 12

future_forecast = (
    final_model_fit.get_forecast(
        steps=forecast_steps
    )
)

forecast_values = (
    future_forecast.predicted_mean
)

forecast_confidence = (
    future_forecast.conf_int()
)


# ---------------------------------------------------------
# Forecast DataFrame
# ---------------------------------------------------------

forecast_df = pd.DataFrame({

    "Month": forecast_values.index,

    "Forecasted_Passengers":
        forecast_values.values

})

forecast_df.to_csv(
    "passenger_forecast.csv",
    index=False
)

print(
    "\nForecast saved successfully:"
)

print(
    "passenger_forecast.csv"
)


# ---------------------------------------------------------
# Forecast Summary
# ---------------------------------------------------------

print("\n==========================================")
print("FORECAST SUMMARY")
print("==========================================")

print(
    "Forecast Period :",
    forecast_values.index[0].strftime("%Y-%m"),
    "to",
    forecast_values.index[-1].strftime("%Y-%m")
)

print(
    "Average Forecasted Passengers :",
    round(
        forecast_values.mean(),
        2
    )
)

print(
    "Maximum Forecasted Passengers :",
    round(
        forecast_values.max(),
        2
    )
)

print(
    "Minimum Forecasted Passengers :",
    round(
        forecast_values.min(),
        2
    )
)

print(
    "\nSTEP 10 COMPLETED SUCCESSFULLY"
)


# =========================================================
# STEP 11 - MODEL EVALUATION
# =========================================================

print("\n==========================================")
print("STEP 11 - MODEL EVALUATION")
print("==========================================")


# Actual values
actual = test_data.values


# Predicted values
predicted = test_forecast_values.values


# Make lengths equal
min_length = min(
    len(actual),
    len(predicted)
)

actual = actual[:min_length]

predicted = predicted[:min_length]


# ---------------------------------------------------------
# MAE
# ---------------------------------------------------------

mae = mean_absolute_error(
    actual,
    predicted
)


# ---------------------------------------------------------
# RMSE
# ---------------------------------------------------------

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predicted
    )
)


# ---------------------------------------------------------
# MAPE
# ---------------------------------------------------------

non_zero = actual != 0

mape = np.mean(

    np.abs(

        (
            actual[non_zero]
            -
            predicted[non_zero]
        )
        /
        actual[non_zero]

    )

) * 100


# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------

print("\nMODEL EVALUATION RESULTS")

print("------------------------------------------")

print(
    "MAE  :",
    round(mae, 2)
)

print(
    "RMSE :",
    round(rmse, 2)
)

print(
    "MAPE :",
    round(mape, 2),
    "%"
)

print("------------------------------------------")


if mape < 10:

    print(
        "Forecast Accuracy : EXCELLENT"
    )

elif mape < 20:

    print(
        "Forecast Accuracy : GOOD"
    )

elif mape < 50:

    print(
        "Forecast Accuracy : REASONABLE"
    )

else:

    print(
        "Forecast Accuracy : NEEDS IMPROVEMENT"
    )


# ---------------------------------------------------------
# Save Evaluation
# ---------------------------------------------------------

evaluation_df = pd.DataFrame({

    "Metric": [
        "MAE",
        "RMSE",
        "MAPE"
    ],

    "Value": [
        mae,
        rmse,
        mape
    ]

})

evaluation_df.to_csv(
    "model_evaluation.csv",
    index=False
)

print(
    "\nEvaluation results saved as:"
)

print(
    "model_evaluation.csv"
)

print(
    "\nSTEP 11 COMPLETED SUCCESSFULLY"
)


# =========================================================
# STEP 12 - AIRLINE-WISE FORECASTING
# =========================================================

print("\n==========================================")
print("STEP 12 - AIRLINE-WISE FORECASTING")
print("==========================================")

airline_forecasts = []

forecast_steps = 12


# ---------------------------------------------------------
# Forecast Each Airline
# ---------------------------------------------------------

for airline in airlines:

    print("\n------------------------------------------")

    print(
        "Forecasting:",
        airline
    )

    print("------------------------------------------")


    # Create monthly time series
    airline_ts = (
        df.set_index("Month")
        [airline]
        .asfreq("MS")
    )


    # Fill missing values
    airline_ts = (
        airline_ts.interpolate()
    )


    # Check data length
    if len(
        airline_ts.dropna()
    ) < 24:

        print(
            "Not enough data for:",
            airline
        )

        continue


    try:

        # -------------------------------------------------
        # Airline SARIMAX Model
        # -------------------------------------------------

        airline_model = SARIMAX(

            airline_ts,

            order=(1, 1, 1),

            seasonal_order=(1, 1, 1, 12),

            enforce_stationarity=False,

            enforce_invertibility=False
        )


        # Fit model
        airline_model_fit = (
            airline_model.fit(
                disp=False
            )
        )


        # -------------------------------------------------
        # Forecast next 12 months
        # -------------------------------------------------

        airline_forecast = (

            airline_model_fit

            .get_forecast(
                steps=forecast_steps
            )

            .predicted_mean

        )


        # -------------------------------------------------
        # Store Results
        # -------------------------------------------------

        for date, value in (
            airline_forecast.items()
        ):

            airline_forecasts.append({

                "Month":
                    date,

                "Airline":
                    airline,

                "Forecasted_Passengers":
                    value

            })


        # -------------------------------------------------
        # Display Summary
        # -------------------------------------------------

        print(
            "Average Forecast :",
            round(
                airline_forecast.mean(),
                2
            )
        )

        print(
            "Maximum Forecast :",
            round(
                airline_forecast.max(),
                2
            )
        )

        print(
            "Minimum Forecast :",
            round(
                airline_forecast.min(),
                2
            )
        )

        print(
            "Forecast completed."
        )


    except Exception as e:

        print(
            "Error for",
            airline,
            ":",
            e
        )


# =========================================================
# SAVE AIRLINE-WISE FORECAST
# =========================================================

airline_forecast_df = pd.DataFrame(
    airline_forecasts
)


airline_forecast_df.to_csv(

    "airline_wise_forecast.csv",

    index=False

)


print("\n==========================================")
print("AIRLINE-WISE FORECAST SAVED")
print("==========================================")

print(
    "File:",
    "airline_wise_forecast.csv"
)

print(
    "Total Forecast Records:",
    len(
        airline_forecast_df
    )
)


print(
    "\nSTEP 12 COMPLETED SUCCESSFULLY"
)

print(
    "=========================================="
)


# =========================================================
# PROJECT STATUS
# =========================================================

print("\n\n")
print("======================================================")
print("       AIR PASSENGER SARIMAX PROJECT")
print("======================================================")

print("STEP 1  : Data Collection              ✓")
print("STEP 2  : Data Preprocessing            ✓")
print("STEP 3  : EDA                           ✓")
print("STEP 4  : Time-Series Visualization     ✓")
print("STEP 5  : ADF Stationarity Test         ✓")
print("STEP 6  : First Differencing            ✓")
print("STEP 7  : Seasonal Differencing         ✓")
print("STEP 8  : ACF & PACF Analysis           ✓")
print("STEP 9  : SARIMAX Model                 ✓")
print("STEP 10 : Forecasting                   ✓")
print("STEP 11 : MAE, RMSE, MAPE               ✓")
print("STEP 12 : Airline-wise Forecasting      ✓")

print("======================================================")
print("       STEPS 1-12 COMPLETED SUCCESSFULLY")
print("======================================================")