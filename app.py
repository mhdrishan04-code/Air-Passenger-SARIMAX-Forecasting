import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Air Passenger Forecast",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="auto"
)


# =========================================================
# MOBILE + WEB CSS
# =========================================================

st.markdown("""
<style>

    /* Main application */
    .block-container {
        padding-top: 1.5rem;
        padding-left: 4%;
        padding-right: 4%;
        max-width: 1400px;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #777;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* KPI cards */
    .kpi-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        min-height: 105px;
        margin-bottom: 10px;
    }

    .kpi-title {
        font-size: 0.85rem;
        opacity: 0.75;
    }

    .kpi-value {
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 8px;
    }

    /* Section titles */
    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1.2rem;
        margin-bottom: 0.7rem;
    }

    /* Mobile */
    @media (max-width: 768px) {

        .block-container {
            padding-left: 4%;
            padding-right: 4%;
            padding-top: 1rem;
        }

        .main-title {
            font-size: 1.55rem;
        }

        .subtitle {
            font-size: 0.85rem;
        }

        .section-title {
            font-size: 1.15rem;
        }

        .kpi-card {
            padding: 14px;
            min-height: 90px;
        }

        .kpi-value {
            font-size: 1.15rem;
        }

        .kpi-title {
            font-size: 0.75rem;
        }

        /* Make dataframe easier to scroll */
        [data-testid="stDataFrame"] {
            font-size: 0.8rem;
        }

    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

try:

    df = pd.read_csv(
        "passengers.csv"
    )

except FileNotFoundError:

    st.error(
        "❌ passengers.csv not found."
    )

    st.stop()


# =========================================================
# LOAD FORECAST
# =========================================================

forecast_file = Path(
    "passenger_forecast.csv"
)

if forecast_file.exists():

    forecast_df = pd.read_csv(
        forecast_file
    )

else:

    forecast_df = pd.DataFrame()


# =========================================================
# LOAD AIRLINE FORECAST
# =========================================================

airline_forecast_file = Path(
    "airline_wise_forecast.csv"
)

if airline_forecast_file.exists():

    airline_forecast_df = pd.read_csv(
        airline_forecast_file
    )

else:

    airline_forecast_df = pd.DataFrame()


# =========================================================
# LOAD EVALUATION
# =========================================================

evaluation_file = Path(
    "model_evaluation.csv"
)

if evaluation_file.exists():

    evaluation_df = pd.read_csv(
        evaluation_file
    )

else:

    evaluation_df = pd.DataFrame()


# =========================================================
# DATA PREPROCESSING
# =========================================================

df["Month"] = pd.to_datetime(
    df["Month"]
)

df = df.sort_values(
    "Month"
)


# =========================================================
# AIRLINES
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


# =========================================================
# TOTAL PASSENGERS
# =========================================================

df["Total_Passengers"] = (
    df[airlines].sum(axis=1)
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">✈️ Air Passenger Forecast</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'SARIMAX International Airline Passenger Forecasting'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Dashboard")

    selected_airline = st.selectbox(
        "Select Airline",
        airlines
    )

    st.divider()

    st.write(
        "**Project:** Air Passenger SARIMAX"
    )

    st.write(
        "**Forecast:** 12 Months"
    )

    st.write(
        "**Model:** SARIMAX"
    )


# =========================================================
# KPI VALUES
# =========================================================

latest_passengers = (
    df["Total_Passengers"].iloc[-1]
)

average_passengers = (
    df["Total_Passengers"].mean()
)

maximum_passengers = (
    df["Total_Passengers"].max()
)

minimum_passengers = (
    df["Total_Passengers"].min()
)


# =========================================================
# KPI SECTION
# =========================================================

st.markdown(
    '<div class="section-title">📊 Passenger Overview</div>',
    unsafe_allow_html=True
)


k1, k2, k3, k4 = st.columns(
    4
)


with k1:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">
                Latest Passengers
            </div>
            <div class="kpi-value">
                {latest_passengers:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with k2:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">
                Average
            </div>
            <div class="kpi-value">
                {average_passengers:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with k3:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">
                Maximum
            </div>
            <div class="kpi-value">
                {maximum_passengers:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with k4:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">
                Minimum
            </div>
            <div class="kpi-value">
                {minimum_passengers:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HISTORICAL TREND
# =========================================================

st.markdown(
    '<div class="section-title">'
    '📈 Historical Passenger Traffic'
    '</div>',
    unsafe_allow_html=True
)


fig_history = px.line(

    df,

    x="Month",

    y="Total_Passengers",

    markers=True

)


fig_history.update_layout(

    xaxis_title="Month",

    yaxis_title="Passengers",

    hovermode="x unified",

    margin=dict(
        l=10,
        r=10,
        t=20,
        b=10
    )
)


st.plotly_chart(
    fig_history,
    use_container_width=True
)


# =========================================================
# SELECTED AIRLINE
# =========================================================

st.markdown(
    f'<div class="section-title">'
    f'✈️ {selected_airline}'
    f'</div>',
    unsafe_allow_html=True
)


airline_data = df[
    [
        "Month",
        selected_airline
    ]
].copy()


fig_airline = px.line(

    airline_data,

    x="Month",

    y=selected_airline,

    markers=True

)


fig_airline.update_layout(

    xaxis_title="Month",

    yaxis_title="Passengers",

    hovermode="x unified",

    margin=dict(
        l=10,
        r=10,
        t=20,
        b=10
    )
)


st.plotly_chart(
    fig_airline,
    use_container_width=True
)


# =========================================================
# FUTURE FORECAST
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🔮 12-Month Forecast'
    '</div>',
    unsafe_allow_html=True
)


if not forecast_df.empty:

    forecast_column = None

    for column in forecast_df.columns:

        if "forecast" in column.lower():

            forecast_column = column

            break


    if forecast_column:

        forecast_df["Month"] = pd.to_datetime(
            forecast_df["Month"]
        )


        fig_forecast = px.line(

            forecast_df,

            x="Month",

            y=forecast_column,

            markers=True

        )


        fig_forecast.update_layout(

            xaxis_title="Month",

            yaxis_title="Forecasted Passengers",

            hovermode="x unified",

            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            )
        )


        st.plotly_chart(
            fig_forecast,
            use_container_width=True
        )


        fa, fb, fc = st.columns(3)


        forecast_average = (
            forecast_df[
                forecast_column
            ].mean()
        )

        forecast_max = (
            forecast_df[
                forecast_column
            ].max()
        )

        forecast_min = (
            forecast_df[
                forecast_column
            ].min()
        )


        fa.metric(
            "Average Forecast",
            f"{forecast_average:,.0f}"
        )

        fb.metric(
            "Maximum Forecast",
            f"{forecast_max:,.0f}"
        )

        fc.metric(
            "Minimum Forecast",
            f"{forecast_min:,.0f}"
        )


# =========================================================
# AIRLINE-WISE FORECAST
# =========================================================

st.markdown(
    '<div class="section-title">'
    '✈️ Airline-wise Forecast'
    '</div>',
    unsafe_allow_html=True
)


if not airline_forecast_df.empty:

    selected_forecast = (
        airline_forecast_df[
            airline_forecast_df["Airline"]
            == selected_airline
        ]
        .copy()
    )


    if not selected_forecast.empty:

        selected_forecast["Month"] = (
            pd.to_datetime(
                selected_forecast["Month"]
            )
        )


        fig_airline_forecast = px.line(

            selected_forecast,

            x="Month",

            y="Forecasted_Passengers",

            markers=True

        )


        fig_airline_forecast.update_layout(

            xaxis_title="Month",

            yaxis_title="Forecasted Passengers",

            hovermode="x unified",

            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            )
        )


        st.plotly_chart(
            fig_airline_forecast,
            use_container_width=True
        )


        with st.expander(
            "📋 View Forecast Table"
        ):

            st.dataframe(
                selected_forecast,
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# MODEL EVALUATION
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🎯 Model Evaluation'
    '</div>',
    unsafe_allow_html=True
)


if not evaluation_df.empty:

    evaluation_values = {}

    for _, row in evaluation_df.iterrows():

        evaluation_values[
            row["Metric"]
        ] = row["Value"]


    mae = evaluation_values.get(
        "MAE",
        0
    )

    rmse = evaluation_values.get(
        "RMSE",
        0
    )

    mape = evaluation_values.get(
        "MAPE",
        0
    )


    e1, e2, e3 = st.columns(3)


    e1.metric(
        "MAE",
        f"{mae:,.2f}"
    )

    e2.metric(
        "RMSE",
        f"{rmse:,.2f}"
    )

    e3.metric(
        "MAPE",
        f"{mape:.2f}%"
    )


    if mape < 10:

        st.success(
            "MAPE: Excellent forecast accuracy"
        )

    elif mape < 20:

        st.info(
            "MAPE: Good forecast accuracy"
        )

    elif mape < 50:

        st.warning(
            "MAPE: Reasonable forecast accuracy"
        )

    else:

        st.error(
            "MAPE: Model needs improvement"
        )


# =========================================================
# FORECAST TABLE
# =========================================================

with st.expander(
    "📋 View Overall Forecast Table"
):

    if not forecast_df.empty:

        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# PROJECT DETAILS
# =========================================================

with st.expander(
    "ℹ️ Project Details"
):

    st.markdown(
        """
        **Project:** Air Passenger Forecasting

        **Model:** SARIMAX

        **Forecast Horizon:** 12 Months

        **Dataset:** International Airline Passenger Data

        **Tech Stack:**
        - Python
        - Pandas
        - NumPy
        - Statsmodels
        - Scikit-learn
        - Plotly
        - Streamlit

        **Workflow:**
        1. Data Collection
        2. Data Preprocessing
        3. EDA
        4. Stationarity Testing
        5. Differencing
        6. Seasonal Differencing
        7. ACF/PACF
        8. SARIMAX
        9. Forecasting
        10. Model Evaluation
        11. Airline-wise Forecasting
        12. Streamlit Dashboard
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "✈️ Air Passenger SARIMAX Forecasting | "
    "Responsive Mobile + Web UI"
)

st.success(
    "✅ Mobile & Web UI Ready"
)
