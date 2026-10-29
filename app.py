from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from src.realtime_weather import get_live_weather
from src.safety_assessment import assess_safety
from src.alert_engine import generate_alert


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CLEAN_DIR = BASE_DIR / "data" / "cleaned"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Transportation Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #08111f;
    }

    [data-testid="stSidebar"] {
        background-color: #0b1628;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
    }

    .metric-card {
        background-color: #101f34;
        border: 1px solid #1e3958;
        border-radius: 16px;
        padding: 18px;
    }

    .alert-card {
        background-color: #111d2e;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .small-text {
        color: #8fa6c1;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    accidents = pd.read_csv(
        CLEAN_DIR / "road_accidents_clean.csv"
    )

    transport = pd.read_csv(
        CLEAN_DIR / "transport_clean.csv"
    )

    weather = pd.read_csv(
        CLEAN_DIR / "weather_clean.csv"
    )

    ml_data = pd.read_csv(
        PROCESSED_DIR / "accident_ml_dataset.csv"
    )

    # --------------------------------------------------------
    # CLEAN TEXT COLUMNS
    # --------------------------------------------------------

    accidents["State"] = (
        accidents["State"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    transport["City Name"] = (
        transport["City Name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    weather["city"] = (
        weather["city"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    ml_data["State"] = (
        ml_data["State"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return accidents, transport, weather, ml_data


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {}

    # Primary model used for current prediction and XAI
    models["Random Forest"] = joblib.load(
        MODELS_DIR / "random_forest.joblib"
    )

    # Hybrid ensemble model
    models["Hybrid Ensemble"] = joblib.load(
        MODELS_DIR / "hybrid_ensemble.joblib"
    )

    return models


# ============================================================
# LOAD PROJECT FILES
# ============================================================

try:

    accidents, transport, weather, ml_data = load_data()

    models = load_models()

    # Random Forest remains the primary prediction model.
    # Existing SHAP/LIME explanations are based on Random Forest.
    model = models["Random Forest"]

except Exception as error:

    st.error(
        "Required project files could not be loaded."
    )

    st.code(str(error))

    st.stop()


# ============================================================
# AVAILABLE LOCATIONS
# ============================================================

states = sorted(
    [
        state
        for state in ml_data["State"].unique()
        if state
    ]
)

cities = sorted(
    [
        city
        for city in weather["city"].unique()
        if city
    ]
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_weather_context(city):

    city_data = weather[
        weather["city"] == city
    ].sort_values("date")

    if city_data.empty:
        return None

    latest = city_data.iloc[-1]

    temperature = float(
        latest["temperature_2m_max"]
    )

    rain = float(
        latest["rain_sum"]
    )

    wind = float(
        latest["wind_speed_10m_max"]
    )

    if rain > 10 or wind > 40:

        status = "Adverse"

    elif rain > 3 or wind > 25:

        status = "Moderate"

    else:

        status = "Normal"

    return {
        "temperature": temperature,
        "rain": rain,
        "wind": wind,
        "status": status
    }


# ============================================================
# REAL-TIME WEATHER
# ============================================================

def get_current_weather(city):

    return get_live_weather(city)


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_model_input(state):

    row = ml_data[
        ml_data["State"] == state
    ].iloc[0].copy()

    # --------------------------------------------------------
    # ORIGINAL ACCIDENT FEATURES
    # --------------------------------------------------------

    accidents_2019 = float(
        row["2019 Accidents"]
    )

    accidents_2020 = float(
        row["2020 Accidents"]
    )

    accidents_2021 = float(
        row["2021 Accidents"]
    )

    accidents_2022 = float(
        row["2022 Accidents"]
    )

    accident_values = [
        accidents_2019,
        accidents_2020,
        accidents_2021,
        accidents_2022
    ]

    # --------------------------------------------------------
    # AVERAGE ACCIDENTS
    # --------------------------------------------------------

    average_accidents = np.mean(
        accident_values
    )

    # --------------------------------------------------------
    # TOTAL ACCIDENTS
    # --------------------------------------------------------

    total_accidents_2019_2022 = np.sum(
        accident_values
    )

    # --------------------------------------------------------
    # GROWTH FROM 2019 TO 2022
    # --------------------------------------------------------

    if accidents_2019 != 0:

        accident_growth_2019_2022 = (
            (
                accidents_2022
                - accidents_2019
            )
            / accidents_2019
        ) * 100

    else:

        accident_growth_2019_2022 = 0

    # --------------------------------------------------------
    # STANDARD DEVIATION / VARIATION
    # --------------------------------------------------------

    accident_variation = np.std(
        accident_values
    )

    # --------------------------------------------------------
    # CREATE MODEL INPUT
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        {
            "2019 Accidents": [
                accidents_2019
            ],

            "2020 Accidents": [
                accidents_2020
            ],

            "2021 Accidents": [
                accidents_2021
            ],

            "2022 Accidents": [
                accidents_2022
            ],

            "average_accidents": [
                average_accidents
            ],

            "total_accidents_2019_2022": [
                total_accidents_2019_2022
            ],

            "accident_growth_2019_2022": [
                accident_growth_2019_2022
            ],

            "accident_variation": [
                accident_variation
            ]
        }
    )

    return input_data


# ============================================================
# ACCIDENT RISK PREDICTION
# ============================================================

def predict_risk(state):

    input_data = create_model_input(
        state
    )

    prediction = model.predict(
        input_data
    )[0]

    probabilities = model.predict_proba(
        input_data
    )[0]

    confidence = max(
        probabilities
    )

    probability_dict = dict(
        zip(
            model.classes_,
            probabilities
        )
    )

    return (
        prediction,
        confidence,
        probability_dict
    )


# ============================================================
# MODEL-SPECIFIC RISK PREDICTION
# ============================================================

def predict_risk_with_model(
    state,
    model_name
):

    input_data = create_model_input(
        state
    )

    selected_model = models[
        model_name
    ]

    prediction = selected_model.predict(
        input_data
    )[0]

    probabilities = selected_model.predict_proba(
        input_data
    )[0]

    confidence = max(
        probabilities
    )

    probability_dict = dict(
        zip(
            selected_model.classes_,
            probabilities
        )
    )

    return (
        prediction,
        confidence,
        probability_dict
    )


# ============================================================
# RISK SCORE
# ============================================================

def risk_score(level):

    if level == "Low":
        return 25

    if level == "Medium":
        return 55

    if level == "High":
        return 85

    return 50


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🚦 SMART TRANSPORT"
)

st.sidebar.caption(
    "Mobility Intelligence & Predictive Safety"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Live Command Center",
        "Smart Route",
        "Accident Risk",
        "Weather Intelligence",
        "Transport Intelligence",
        "Explainable AI",
        "Model Performance",
        "Analytics & Decision Support"
    ]
)


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "Smart Transportation Intelligence"
)

st.caption(
    "Current mobility awareness • predictive safety • "
    "explainable AI • intelligent alerts"
)


# ============================================================
# 1. LIVE COMMAND CENTER
# ============================================================

if page == "Live Command Center":

    st.subheader(
        "🚦 Smart Transportation Intelligence"
    )

    st.caption(
        "Real-time transportation safety monitoring "
        "using historical accident analytics and "
        "current weather conditions."
    )

    # --------------------------------------------------------
    # MONITORING INPUT
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        selected_city = st.selectbox(
            "📍 Monitoring City",
            cities,
            key="command_city"
        )

    with col2:

        destination = st.text_input(
            "🎯 Destination",
            placeholder="Enter destination",
            key="command_destination"
        )

    with col3:

        monitoring_state = st.selectbox(
            "🗺️ Risk Analysis Region",
            states,
            key="command_state"
        )

    # --------------------------------------------------------
    # HISTORICAL ACCIDENT MODEL
    # --------------------------------------------------------

    risk, confidence, probabilities = predict_risk(
        monitoring_state
    )

    # --------------------------------------------------------
    # LIVE WEATHER
    # --------------------------------------------------------

    live_weather = get_live_weather(
        selected_city
    )

    # --------------------------------------------------------
    # SAFETY ASSESSMENT
    # --------------------------------------------------------

    safety_result = assess_safety(
        risk,
        live_weather
    )

    alert = generate_alert(
        safety_result
    )

    # --------------------------------------------------------
    # JOURNEY STATUS
    # --------------------------------------------------------

    if destination:

        st.info(
            f"🚗 Monitoring journey from "
            f"**{selected_city}** to "
            f"**{destination}**"
        )

    else:

        st.info(
            f"📡 Currently monitoring "
            f"**{selected_city}**"
        )

    st.divider()

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    st.markdown(
        "### 📊 Current Transportation Status"
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Accident Risk",
        risk
    )

    k2.metric(
        "AI Confidence",
        f"{confidence:.0%}"
    )

    k3.metric(
        "Safety Level",
        safety_result["level"]
    )

    if live_weather:

        k4.metric(
            "Weather",
            live_weather["status"]
        )

    else:

        k4.metric(
            "Weather",
            "Unavailable"
        )

    # --------------------------------------------------------
    # SAFETY SCORE
    # --------------------------------------------------------

    st.markdown(
        "### 🛡️ Current Safety Assessment"
    )

    score_col1, score_col2 = st.columns(
        [1, 2]
    )

    with score_col1:

        st.metric(
            "Safety Risk Score",
            f"{safety_result['score']}/100"
        )

    with score_col2:

        if safety_result["level"] == "High":

            st.error(
                "🚨 HIGH SAFETY RISK\n\n"
                + safety_result["alert"]
            )

        elif safety_result["level"] == "Medium":

            st.warning(
                "⚠️ MODERATE SAFETY RISK\n\n"
                + safety_result["alert"]
            )

        else:

            st.success(
                "✅ LOW SAFETY RISK\n\n"
                + safety_result["alert"]
            )

    # --------------------------------------------------------
    # LIVE WEATHER
    # --------------------------------------------------------

    st.markdown(
        "### 🌦️ Live Weather Intelligence"
    )

    if live_weather:

        w1, w2, w3, w4, w5 = st.columns(5)

        w1.metric(
            "Temperature",
            f"{live_weather['temperature']:.1f} °C"
        )

        w2.metric(
            "Humidity",
            f"{live_weather['humidity']:.0f}%"
        )

        w3.metric(
            "Rain",
            f"{live_weather['rain']:.1f} mm"
        )

        w4.metric(
            "Wind",
            f"{live_weather['wind']:.1f} km/h"
        )

        w5.metric(
            "Condition",
            live_weather["condition"]
        )

    else:

        st.warning(
            "Live weather information is currently unavailable."
        )

    # --------------------------------------------------------
    # RISK PROBABILITY
    # --------------------------------------------------------

    st.markdown(
        "### 📈 Accident Risk Probability"
    )

    probability_df = pd.DataFrame(
        {
            "Risk Level": list(
                probabilities.keys()
            ),

            "Probability": [
                float(value) * 100
                for value in probabilities.values()
            ]
        }
    )

    fig_probability = px.bar(
        probability_df,
        x="Risk Level",
        y="Probability",
        text="Probability",
        title="Historical Accident-Risk Prediction"
    )

    fig_probability.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig_probability.update_yaxes(
        title="Probability (%)",
        range=[0, 100]
    )

    st.plotly_chart(
        fig_probability,
        use_container_width=True
    )

# ============================================================
# 2. SMART ROUTE
# ============================================================

elif page == "Smart Route":

    st.subheader(
        "🗺️ Smart Route Safety"
    )

    st.caption(
        "Regional route-safety assessment using historical "
        "accident risk and current weather conditions."
    )

    # --------------------------------------------------------
    # JOURNEY INPUT
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        current_location = st.selectbox(
            "📍 Current Location",
            cities,
            key="route_current_location"
        )

    with col2:

        destination = st.text_input(
            "🎯 Destination",
            placeholder="Enter destination"
        )

    monitoring_state = st.selectbox(
        "AI Risk Analysis Region",
        states,
        key="route_monitoring_state"
    )

    # --------------------------------------------------------
    # HISTORICAL ACCIDENT RISK
    # --------------------------------------------------------

    risk, confidence, probabilities = predict_risk(
        monitoring_state
    )

    # --------------------------------------------------------
    # LIVE WEATHER
    # --------------------------------------------------------

    live_weather = get_current_weather(
        current_location
    )

    # --------------------------------------------------------
    # SAFETY ASSESSMENT
    # --------------------------------------------------------

    safety_result = None
    alert = None

    if live_weather:

        safety_result = assess_safety(
            risk,
            live_weather
        )

        alert = generate_alert(
            safety_result
        )

    # --------------------------------------------------------
    # JOURNEY SUMMARY
    # --------------------------------------------------------

    if destination:

        st.info(
            f"🚗 Monitoring journey from "
            f"**{current_location}** to "
            f"**{destination}**."
        )

    else:

        st.info(
            "Enter a destination to create a journey "
            "safety assessment."
        )

    # --------------------------------------------------------
    # CURRENT CONDITIONS
    # --------------------------------------------------------

    st.markdown(
        "### 📡 Current Journey Conditions"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accident Risk",
        risk
    )

    col2.metric(
        "AI Confidence",
        f"{confidence:.0%}"
    )

    if live_weather:

        col3.metric(
            "Temperature",
            f"{live_weather['temperature']:.1f} °C"
        )

        col4.metric(
            "Weather",
            live_weather["status"]
        )

    else:

        col3.metric(
            "Temperature",
            "Unavailable"
        )

        col4.metric(
            "Weather",
            "Unavailable"
        )

    # --------------------------------------------------------
    # SAFETY SCORE
    # --------------------------------------------------------

    if safety_result:

        st.markdown(
            "### 🛡️ Journey Safety Assessment"
        )

        score_col1, score_col2 = st.columns(2)

        score_col1.metric(
            "Safety Risk Score",
            f"{safety_result['score']}/100"
        )

        score_col2.metric(
            "Safety Level",
            safety_result["level"]
        )

        # ----------------------------------------------------
        # ALERT
        # ----------------------------------------------------

        if alert:

            if alert["type"] == "high":

                st.error(
                    f"{alert['title']}\n\n"
                    f"{alert['message']}"
                )

            elif alert["type"] == "medium":

                st.warning(
                    f"{alert['title']}\n\n"
                    f"{alert['message']}"
                )

            elif alert["type"] == "low":

                st.success(
                    f"{alert['title']}\n\n"
                    f"{alert['message']}"
                )

            else:

                st.info(
                    f"{alert['title']}\n\n"
                    f"{alert['message']}"
                )

    else:

        st.warning(
            "Current weather information is unavailable, "
            "so the combined journey safety assessment "
            "cannot be generated."
        )

    # --------------------------------------------------------
    # RISK COMPONENTS
    # --------------------------------------------------------

    st.markdown(
        "### 📊 Risk Components"
    )

    accident_component = risk_score(
        risk
    )

    weather_component = 0

    if live_weather:

        if live_weather["status"] == "Adverse":

            weather_component = 20

        elif live_weather["status"] == "Moderate":

            weather_component = 10

    component_data = pd.DataFrame(
        {
            "Component": [
                "Historical Accident Risk",
                "Current Weather Contribution"
            ],

            "Risk Contribution": [
                accident_component,
                weather_component
            ]
        }
    )

    fig = px.bar(
        component_data,
        x="Component",
        y="Risk Contribution",
        range_y=[0, 100],
        text="Risk Contribution",
        title="Journey Risk Components"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# 3. ACCIDENT RISK
# ============================================================

elif page == "Accident Risk":

    st.subheader(
        "🚨 Accident Risk Analysis"
    )

    monitoring_state = st.selectbox(
        "📍 Monitored Location",
        states
    )

    risk, confidence, probabilities = predict_risk(
        monitoring_state
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Accident Risk Signal",
        risk
    )

    col2.metric(
        "Prediction Confidence",
        f"{confidence:.0%}"
    )

    probability_data = pd.DataFrame(
        {
            "Risk Level": list(
                probabilities.keys()
            ),

            "Probability": [
                value * 100
                for value in probabilities.values()
            ]
        }
    )

    fig = px.bar(
        probability_data,
        x="Risk Level",
        y="Probability",
        text="Probability",
        title="AI Risk Probability"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_yaxes(
        range=[0, 100]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        "### 🔔 Safety Notification"
    )

    if risk == "High":

        st.error(
            "⚠️ High accident-risk signal detected "
            "for the monitored region."
        )

    elif risk == "Medium":

        st.warning(
            "⚠️ Moderate accident-risk signal detected "
            "for the monitored region."
        )

    else:

        st.success(
            "✅ Lower accident-risk signal detected "
            "under the current model."
        )



# ============================================================
# 4. WEATHER INTELLIGENCE
# ============================================================

elif page == "Weather Intelligence":

    st.subheader(
        "🌦️ Weather Intelligence"
    )

    city = st.selectbox(
        "📍 Monitored Location",
        cities
    )

    # --------------------------------------------------------
    # LIVE WEATHER
    # --------------------------------------------------------

    live_weather = get_current_weather(
        city
    )

    if live_weather:

        st.markdown(
            "### 🟢 Live Weather Conditions"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Temperature",
            f"{live_weather['temperature']:.1f} °C"
        )

        col2.metric(
            "Rain",
            f"{live_weather['rain']:.1f} mm"
        )

        col3.metric(
            "Wind",
            f"{live_weather['wind']:.1f} km/h"
        )

        col4.metric(
            "Condition",
            live_weather["condition"]
        )

        if live_weather["status"] == "Adverse":

            st.error(
                "🌧️ LIVE WEATHER ALERT: "
                "Adverse weather conditions detected."
            )

        elif live_weather["status"] == "Moderate":

            st.warning(
                "🌦️ LIVE WEATHER CAUTION: "
                "Moderate weather conditions detected."
            )

        else:

            st.success(
                "☀️ LIVE WEATHER: "
                "No elevated weather-risk signal detected."
            )

    else:

        st.warning(
            "Live weather data is currently unavailable."
        )

    # --------------------------------------------------------
    # HISTORICAL WEATHER CONTEXT
    # --------------------------------------------------------

    city_weather = weather[
        weather["city"] == city
    ].sort_values("date")

    if not city_weather.empty:

        recent = city_weather.tail(
            14
        ).copy()

        recent["Weather Risk Index"] = (
            recent["rain_sum"].clip(
                lower=0
            ) * 4
            +
            recent[
                "wind_speed_10m_max"
            ].clip(
                lower=0
            ) * 0.8
        )

        fig = px.line(
            recent,
            x="date",
            y="Weather Risk Index",
            markers=True,
            title="Recent Historical Weather Risk Signal"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.caption(
        "Live weather is retrieved from the real-time weather "
        "service. Historical weather data is used only for "
        "context and trend analysis."
    )


# ============================================================
# 5. TRANSPORT INTELLIGENCE
# ============================================================

elif page == "Transport Intelligence":

    st.subheader(
        "🚌 Transport Intelligence"
    )

    transport_summary = transport.groupby(
        "City Name",
        as_index=False
    ).agg(
        {
            "No. of buses of that type": "sum",
            "No. of bus terminals": "max",
            "No. of bus stands": "max",
            "No. of bus stops": "max"
        }
    )

    transport_cities = sorted(
        transport_summary["City Name"].unique()
    )

    city = st.selectbox(
        "📍 Monitored Transport Area",
        transport_cities
    )

    row = transport_summary[
        transport_summary["City Name"] == city
    ].iloc[0]

    buses = float(
        row["No. of buses of that type"]
    )

    stops = float(
        row["No. of bus stops"]
    )

    # --------------------------------------------------------
    # DERIVED TRANSPORTATION LOAD SIGNAL
    # --------------------------------------------------------

    load_signal = min(
        100,
        (buses / max(stops, 1)) * 100
    )

    if load_signal > 80:

        transport_risk = "High"

    elif load_signal > 45:

        transport_risk = "Medium"

    else:

        transport_risk = "Low"

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Network Status",
        "MONITORED"
    )

    col2.metric(
        "Load Signal",
        f"{load_signal:.0f}%"
    )

    col3.metric(
        "Transport Risk",
        transport_risk
    )

    # --------------------------------------------------------
    # TRANSPORT RISK SIGNAL
    # --------------------------------------------------------

    transport_chart = pd.DataFrame(
        {
            "Period": [
                "Current Dataset Signal"
            ],

            "Risk": [
                risk_score(
                    transport_risk
                )
            ]
        }
    )

    fig = px.bar(
        transport_chart,
        x="Period",
        y="Risk",
        range_y=[0, 100],
        text="Risk",
        title="Transport Risk Signal"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if transport_risk == "High":

        st.error(
            "🚦 Elevated transport-load signal detected."
        )

    elif transport_risk == "Medium":

        st.warning(
            "🚦 Moderate transport-load signal detected."
        )

    else:

        st.success(
            "🚌 Lower transport-risk signal detected."
        )



# ============================================================
# 6. EXPLAINABLE AI
# ============================================================

elif page == "Explainable AI":

    st.subheader(
        "🧠 Explainable AI"
    )

    st.caption(
        "Understanding which historical accident patterns "
        "influence the machine-learning prediction."
    )

    # --------------------------------------------------------
    # STATE SELECTION
    # --------------------------------------------------------

    selected_state = st.selectbox(
        "Select Region for Explanation",
        states,
        key="xai_state"
    )

    risk, confidence, probabilities = predict_risk(
        selected_state
    )

    # --------------------------------------------------------
    # PREDICTION SUMMARY
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Risk",
        risk
    )

    col2.metric(
        "Model Confidence",
        f"{confidence:.0%}"
    )

    if risk == "High":

        risk_text = "Elevated historical accident signal"

    elif risk == "Medium":

        risk_text = "Moderate historical accident signal"

    else:

        risk_text = "Lower historical accident signal"

    col3.metric(
        "Interpretation",
        risk_text
    )

    st.divider()

    # --------------------------------------------------------
    # RISK PROBABILITY
    # --------------------------------------------------------

    st.markdown(
        "### 🎯 Risk Probability"
    )

    probability_df = pd.DataFrame(
        {
            "Risk Level": list(
                probabilities.keys()
            ),

            "Probability": [
                float(value) * 100
                for value in probabilities.values()
            ]
        }
    )

    probability_df = probability_df.sort_values(
        "Probability",
        ascending=False
    )

    fig_probability = px.bar(
        probability_df,
        x="Risk Level",
        y="Probability",
        text="Probability",
        title="Model Probability for Each Risk Level"
    )

    fig_probability.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig_probability.update_yaxes(
        title="Probability (%)",
        range=[0, 100]
    )

    st.plotly_chart(
        fig_probability,
        use_container_width=True
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown(
        "### 📊 Global Feature Importance"
    )

    st.write(
        "This chart shows which historical accident features "
        "have the greatest overall influence on the trained "
        "machine-learning model."
    )

    try:

        importance_df = pd.read_csv(
            RESULTS_DIR / "feature_importance.csv"
        )

        feature_column = None
        importance_column = None

        for column in importance_df.columns:

            if column.lower() in [
                "feature",
                "features",
                "feature_name"
            ]:

                feature_column = column

            if column.lower() in [
                "importance",
                "feature importance",
                "importance_score"
            ]:

                importance_column = column

        if (
            feature_column is not None
            and importance_column is not None
        ):

            importance_df = importance_df[
                [
                    feature_column,
                    importance_column
                ]
            ].copy()

            importance_df.columns = [
                "Feature",
                "Importance"
            ]

            importance_df["Importance"] = pd.to_numeric(
                importance_df["Importance"],
                errors="coerce"
            )

            importance_df = (
                importance_df
                .dropna()
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .head(10)
                .sort_values(
                    "Importance",
                    ascending=True
                )
            )

            fig_importance = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                text="Importance",
                title="Top Historical Factors Influencing Risk"
            )

            fig_importance.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside"
            )

            fig_importance.update_xaxes(
                title="Importance"
            )

            fig_importance.update_yaxes(
                title="Feature"
            )

            st.plotly_chart(
                fig_importance,
                use_container_width=True
            )

            with st.expander(
                "View exact feature-importance values"
            ):

                st.dataframe(
                    importance_df.sort_values(
                        "Importance",
                        ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.warning(
                "The feature importance file does not contain "
                "the expected columns."
            )

    except Exception:

        st.info(
            "Feature importance visualization is currently "
            "unavailable."
        )

    # --------------------------------------------------------
    # SHAP + LIME (LIVE EXPLANATIONS)
    # --------------------------------------------------------

    # The XAI page uses the same Random Forest that produces the
    # historical accident-risk prediction above. SHAP and LIME are
    # calculated for the currently selected region instead of relying
    # on pre-generated CSV files.

    xai_features = [
        "2019 Accidents",
        "2020 Accidents",
        "2021 Accidents",
        "2022 Accidents",
        "average_accidents",
        "total_accidents_2019_2022",
        "accident_growth_2019_2022",
        "accident_variation"
    ]

    xai_input = create_model_input(selected_state)

    # Make sure the input column order is exactly the same as the
    # features used by the trained Random Forest.
    try:
        xai_input = xai_input[xai_features]
    except Exception:
        pass

    predicted_class_index = int(
        np.argmax(
            model.predict_proba(xai_input)[0]
        )
    )

    predicted_class = model.classes_[predicted_class_index]

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    st.markdown(
        "### 🔍 SHAP Explanation"
    )

    st.write(
        "SHAP shows how individual features contribute to "
        "the current Random Forest prediction."
    )

    try:

        import shap

        # SHAP TreeExplainer requires the actual tree model, not a
        # scikit-learn Pipeline. If the saved model is a Pipeline,
        # separate preprocessing from the final Random Forest.
        if hasattr(model, "steps"):
            rf_model = model.steps[-1][1]
            preprocessing_pipeline = model[:-1]
            xai_transformed = preprocessing_pipeline.transform(xai_input)

            try:
                feature_names = list(
                    preprocessing_pipeline.get_feature_names_out(
                        xai_input.columns
                    )
                )
            except Exception:
                feature_names = list(xai_input.columns)
        else:
            rf_model = model
            xai_transformed = xai_input
            feature_names = list(xai_input.columns)

        explainer = shap.TreeExplainer(rf_model)
        shap_result = explainer.shap_values(xai_transformed)

        # SHAP has returned different structures across versions:
        # list[class] x samples x features, or
        # samples x features x classes.
        if isinstance(shap_result, list):

            shap_values_for_class = np.asarray(
                shap_result[predicted_class_index]
            )[0]

        else:

            shap_array = np.asarray(shap_result)

            if shap_array.ndim == 3:
                # samples x features x classes
                shap_values_for_class = shap_array[
                    0,
                    :,
                    predicted_class_index
                ]

            elif shap_array.ndim == 2:
                shap_values_for_class = shap_array[0]

            else:
                shap_values_for_class = shap_array.reshape(-1)

        shap_values_for_class = np.asarray(
            shap_values_for_class,
            dtype=float
        ).reshape(-1)

        # Keep feature names aligned with the transformed SHAP matrix.
        if len(feature_names) != len(shap_values_for_class):
            feature_names = [
                f"Feature {i + 1}"
                for i in range(len(shap_values_for_class))
            ]

        shap_plot_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "SHAP Value": shap_values_for_class
            }
        )

        shap_plot_df["Absolute SHAP"] = (
            shap_plot_df["SHAP Value"].abs()
        )

        shap_plot_df = (
            shap_plot_df
            .sort_values(
                "Absolute SHAP",
                ascending=False
            )
            .head(10)
            .sort_values(
                "SHAP Value",
                ascending=True
            )
        )

        # Explainability metrics
        shap_mean_abs = float(
            shap_plot_df["Absolute SHAP"].mean()
        )

        shap_max_abs = float(
            shap_plot_df["Absolute SHAP"].max()
        )

        shap_positive = int(
            (shap_plot_df["SHAP Value"] > 0).sum()
        )

        shap_negative = int(
            (shap_plot_df["SHAP Value"] < 0).sum()
        )

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Mean |SHAP|",
            f"{shap_mean_abs:.4f}"
        )

        s2.metric(
            "Max |SHAP|",
            f"{shap_max_abs:.4f}"
        )

        s3.metric(
            "Positive Features",
            shap_positive
        )

        s4.metric(
            "Negative Features",
            shap_negative
        )

        fig_shap = px.bar(
            shap_plot_df,
            x="SHAP Value",
            y="Feature",
            orientation="h",
            text="SHAP Value",
            title=(
                "SHAP Contributions for "
                f"{selected_state} → {predicted_class}"
            )
        )

        fig_shap.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )

        fig_shap.update_xaxes(
            title="SHAP Contribution"
        )

        fig_shap.update_yaxes(
            title="Feature"
        )

        st.plotly_chart(
            fig_shap,
            use_container_width=True
        )

        with st.expander("View exact SHAP values"):

            st.dataframe(
                shap_plot_df[
                    [
                        "Feature",
                        "SHAP Value",
                        "Absolute SHAP"
                    ]
                ].sort_values(
                    "Absolute SHAP",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

        st.info(
            "Positive SHAP values push the prediction toward the "
            f"{predicted_class} class, while negative values push "
            "it away from that class. SHAP indicates model "
            "attribution, not causation."
        )

    except ImportError:

        st.error(
            "SHAP is not installed in the active virtual environment. "
            "Run: pip install shap"
        )

    except Exception as error:

        st.error(
            "SHAP explanation could not be generated."
        )

        with st.expander("View SHAP error details"):
            st.code(str(error))

    # --------------------------------------------------------
    # LIME
    # --------------------------------------------------------

    st.markdown(
        "### 🧩 LIME Local Explanation"
    )

    st.write(
        "LIME builds a local interpretable approximation around "
        "the selected region and shows which feature conditions "
        "support or oppose the prediction."
    )

    try:

        from lime.lime_tabular import LimeTabularExplainer

        # LIME needs the training distribution. The historical ML
        # dataset provides the same eight accident-risk features.
        lime_training = ml_data[xai_features].copy()
        lime_training = lime_training.replace(
            [np.inf, -np.inf],
            np.nan
        )
        lime_training = lime_training.fillna(
            lime_training.median(numeric_only=True)
        )

        lime_training_values = lime_training.to_numpy(
            dtype=float
        )

        lime_explainer = LimeTabularExplainer(
            lime_training_values,
            feature_names=xai_features,
            class_names=[str(c) for c in model.classes_],
            mode="classification",
            discretize_continuous=True,
            random_state=42
        )

        lime_exp = lime_explainer.explain_instance(
            xai_input.iloc[0].to_numpy(dtype=float),
            model.predict_proba,
            num_features=min(10, len(xai_features)),
            top_labels=1
        )

        lime_rows = lime_exp.as_list(
            label=predicted_class_index
        )

        lime_plot_df = pd.DataFrame(
            lime_rows,
            columns=[
                "Feature / Condition",
                "LIME Weight"
            ]
        )

        lime_plot_df["LIME Weight"] = pd.to_numeric(
            lime_plot_df["LIME Weight"],
            errors="coerce"
        )

        lime_plot_df = lime_plot_df.dropna()

        lime_mean_abs = float(
            lime_plot_df["LIME Weight"].abs().mean()
        )

        lime_max_abs = float(
            lime_plot_df["LIME Weight"].abs().max()
        )

        lime_positive = int(
            (lime_plot_df["LIME Weight"] > 0).sum()
        )

        lime_negative = int(
            (lime_plot_df["LIME Weight"] < 0).sum()
        )

        l1, l2, l3, l4 = st.columns(4)

        l1.metric(
            "Mean |LIME|",
            f"{lime_mean_abs:.4f}"
        )

        l2.metric(
            "Max |LIME|",
            f"{lime_max_abs:.4f}"
        )

        l3.metric(
            "Positive Conditions",
            lime_positive
        )

        l4.metric(
            "Negative Conditions",
            lime_negative
        )

        lime_plot_df = (
            lime_plot_df
            .sort_values(
                "LIME Weight",
                ascending=True
            )
        )

        fig_lime = px.bar(
            lime_plot_df,
            x="LIME Weight",
            y="Feature / Condition",
            orientation="h",
            text="LIME Weight",
            title=(
                "LIME Local Contributions for "
                f"{selected_state} → {predicted_class}"
            )
        )

        fig_lime.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )

        fig_lime.update_xaxes(
            title="LIME Weight"
        )

        fig_lime.update_yaxes(
            title="Feature / Condition"
        )

        st.plotly_chart(
            fig_lime,
            use_container_width=True
        )

        with st.expander("View exact LIME conditions"):

            st.dataframe(
                lime_plot_df,
                use_container_width=True,
                hide_index=True
            )

        st.info(
            "Positive LIME weights support the selected prediction "
            "while negative weights oppose it. LIME is a local "
            "approximation and should not be interpreted as causal evidence."
        )

    except ImportError:

        st.error(
            "LIME is not installed in the active virtual environment. "
            "Run: pip install lime"
        )

    except Exception as error:

        st.error(
            "LIME explanation could not be generated."
        )

        with st.expander("View LIME error details"):
            st.code(str(error))

    st.caption(
        "SHAP and LIME are calculated for the current Random Forest "
        "historical accident-risk prediction. They explain model "
        "behavior and should not be interpreted as causal effects."
    )

    # ========================================================
    # NEW: MULTIMODAL FEATURE IMPORTANCE
    # ========================================================

    st.markdown(
        "### 🌐 Multimodal Feature Importance"
    )

    st.write(
        "This analysis shows the relative importance of "
        "features when accident, transport and weather "
        "information are analyzed together."
    )

    try:

        multimodal_file = (
            RESULTS_DIR / "multimodal_feature_importance.csv"
        )

        multimodal_df = pd.read_csv(
            multimodal_file
        )

        feature_col = None
        importance_col = None

        for column in multimodal_df.columns:

            lower_name = column.lower().strip()

            if (
                feature_col is None
                and (
                    lower_name == "feature"
                    or "feature" in lower_name
                )
            ):

                feature_col = column

            if (
                importance_col is None
                and "importance" in lower_name
            ):

                importance_col = column

        if (
            feature_col is not None
            and importance_col is not None
        ):

            multimodal_plot_df = multimodal_df[
                [
                    feature_col,
                    importance_col
                ]
            ].copy()

            multimodal_plot_df.columns = [
                "Feature",
                "Importance"
            ]

            multimodal_plot_df["Importance"] = pd.to_numeric(
                multimodal_plot_df["Importance"],
                errors="coerce"
            )

            multimodal_plot_df = (
                multimodal_plot_df
                .dropna()
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .head(15)
                .sort_values(
                    "Importance",
                    ascending=True
                )
            )

            fig_multimodal = px.bar(
                multimodal_plot_df,
                x="Importance",
                y="Feature",
                orientation="h",
                text="Importance",
                title="Top Multimodal Features"
            )

            fig_multimodal.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside"
            )

            fig_multimodal.update_xaxes(
                title="Feature Importance"
            )

            fig_multimodal.update_yaxes(
                title="Feature"
            )

            st.plotly_chart(
                fig_multimodal,
                use_container_width=True
            )

            with st.expander(
                "View multimodal feature-importance values"
            ):

                st.dataframe(
                    multimodal_plot_df.sort_values(
                        "Importance",
                        ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.warning(
                "The multimodal feature-importance file "
                "does not contain the expected columns."
            )

    except Exception:

        st.info(
            "Multimodal feature importance is currently "
            "unavailable."
        )

    # ========================================================
    # NEW: PERMUTATION IMPORTANCE
    # ========================================================

    st.markdown(
        "### 🔄 Permutation Importance"
    )

    st.write(
        "Permutation importance measures how much model "
        "performance changes when a feature is randomly "
        "permuted. Larger values indicate stronger "
        "predictive dependence."
    )

    try:

        permutation_file = (
            RESULTS_DIR / "permutation_importance.csv"
        )

        permutation_df = pd.read_csv(
            permutation_file
        )

        feature_col = None
        mean_col = None
        std_col = None

        for column in permutation_df.columns:

            lower_name = column.lower().strip()

            if (
                feature_col is None
                and (
                    lower_name == "feature"
                    or "feature" in lower_name
                )
            ):

                feature_col = column

            if (
                mean_col is None
                and (
                    "permutation importance mean"
                    in lower_name
                    or (
                        "permutation" in lower_name
                        and "mean" in lower_name
                    )
                )
            ):

                mean_col = column

            if (
                std_col is None
                and (
                    "permutation importance std"
                    in lower_name
                    or (
                        "permutation" in lower_name
                        and "std" in lower_name
                    )
                )
            ):

                std_col = column

        if (
            feature_col is not None
            and mean_col is not None
        ):

            permutation_plot_df = permutation_df[
                [
                    feature_col,
                    mean_col
                ]
            ].copy()

            permutation_plot_df.columns = [
                "Feature",
                "Permutation Importance"
            ]

            permutation_plot_df[
                "Permutation Importance"
            ] = pd.to_numeric(
                permutation_plot_df[
                    "Permutation Importance"
                ],
                errors="coerce"
            )

            permutation_plot_df = (
                permutation_plot_df
                .dropna()
                .sort_values(
                    "Permutation Importance",
                    ascending=False
                )
                .head(15)
                .sort_values(
                    "Permutation Importance",
                    ascending=True
                )
            )

            if std_col is not None:

                std_values = permutation_df[
                    [
                        feature_col,
                        std_col
                    ]
                ].copy()

                std_values.columns = [
                    "Feature",
                    "Std"
                ]

                std_values["Std"] = pd.to_numeric(
                    std_values["Std"],
                    errors="coerce"
                )

                permutation_plot_df = (
                    permutation_plot_df
                    .merge(
                        std_values,
                        on="Feature",
                        how="left"
                    )
                )

                fig_permutation = px.bar(
                    permutation_plot_df,
                    x="Permutation Importance",
                    y="Feature",
                    orientation="h",
                    error_x="Std",
                    text="Permutation Importance",
                    title="Multimodal Permutation Importance"
                )

            else:

                fig_permutation = px.bar(
                    permutation_plot_df,
                    x="Permutation Importance",
                    y="Feature",
                    orientation="h",
                    text="Permutation Importance",
                    title="Multimodal Permutation Importance"
                )

            fig_permutation.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside"
            )

            fig_permutation.update_xaxes(
                title="Permutation Importance"
            )

            fig_permutation.update_yaxes(
                title="Feature"
            )

            st.plotly_chart(
                fig_permutation,
                use_container_width=True
            )

            st.caption(
                "Permutation importance indicates predictive "
                "dependence between a feature and model "
                "performance. It should not be interpreted "
                "as evidence of causation."
            )

            with st.expander(
                "View permutation-importance values"
            ):

                st.dataframe(
                    permutation_plot_df.sort_values(
                        "Permutation Importance",
                        ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.warning(
                "The permutation-importance file does not "
                "contain the expected columns."
            )

    except Exception:

        st.info(
            "Permutation importance results are currently "
            "unavailable."
        )


# ============================================================
# 7. ANALYTICS & DECISION SUPPORT
# ============================================================

elif page == "Analytics & Decision Support":

    st.subheader(
        "📊 Analytics, Storytelling & Decision Support"
    )

    st.caption(
        "Statistical analysis, multidimensional visualization, "
        "data storytelling, responsible AI and actionable decision support."
    )

    # --------------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # --------------------------------------------------------

    st.markdown("### 1. Statistical Analytics")
    st.write(
        "Descriptive statistics summarize the historical accident data "
        "before predictive modelling. The 2022–2023 paired comparison "
        "uses the same states, so it is suitable for a paired test."
    )

    accident_years = [
        "2019 Accidents",
        "2020 Accidents",
        "2021 Accidents",
        "2022 Accidents",
        "2023 Accidents"
    ]

    available_years = [
        c for c in accident_years
        if c in accidents.columns
    ]

    if available_years:
        stats_df = accidents[available_years].apply(
            pd.to_numeric, errors="coerce"
        ).describe().T
        stats_df = stats_df[[
            "mean", "50%", "std", "min", "max"
        ]].rename(columns={
            "50%": "median"
        })

        st.dataframe(
            stats_df.round(2),
            use_container_width=True
        )

        # Paired t-test: 2022 vs 2023, if scipy is available.
        if {"2022 Accidents", "2023 Accidents"}.issubset(
            accidents.columns
        ):
            try:
                from scipy.stats import ttest_rel

                paired = accidents[[
                    "2022 Accidents", "2023 Accidents"
                ]].apply(pd.to_numeric, errors="coerce").dropna()

                test_result = ttest_rel(
                    paired["2022 Accidents"],
                    paired["2023 Accidents"]
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Paired t-statistic", f"{test_result.statistic:.3f}")
                c2.metric("p-value", f"{test_result.pvalue:.4f}")
                c3.metric("States compared", len(paired))

                if test_result.pvalue < 0.05:
                    st.info(
                        "At the 5% significance level, the paired test "
                        "indicates a statistically significant difference "
                        "between the 2022 and 2023 state-level accident counts."
                    )
                else:
                    st.info(
                        "At the 5% significance level, the paired test does "
                        "not indicate a statistically significant difference "
                        "between the 2022 and 2023 state-level accident counts."
                    )
            except ImportError:
                st.info(
                    "Install scipy to display the paired t-test: pip install scipy"
                )

    # --------------------------------------------------------
    # MULTIDIMENSIONAL VISUALIZATION
    # --------------------------------------------------------

    st.markdown("### 2. Multi-dimensional Visual Analytics")

    if len(available_years) >= 2:

        corr_df = accidents[available_years].apply(
            pd.to_numeric, errors="coerce"
        ).corr()

        fig_corr = px.imshow(
            corr_df,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Matrix of Annual Accident Counts"
        )
        st.plotly_chart(
            fig_corr,
            use_container_width=True
        )

    if {"2022 Accidents", "2023 Accidents"}.issubset(
        accidents.columns
    ):
        scatter_df = accidents[[
            "State", "2022 Accidents", "2023 Accidents"
        ]].copy()
        scatter_df["2022 Accidents"] = pd.to_numeric(
            scatter_df["2022 Accidents"], errors="coerce"
        )
        scatter_df["2023 Accidents"] = pd.to_numeric(
            scatter_df["2023 Accidents"], errors="coerce"
        )
        scatter_df = scatter_df.dropna()

        fig_scatter = px.scatter(
            scatter_df,
            x="2022 Accidents",
            y="2023 Accidents",
            hover_name="State",
            title="State-level Accident Change: 2022 vs 2023",
            labels={
                "2022 Accidents": "2022 accidents",
                "2023 Accidents": "2023 accidents"
            }
        )
        st.plotly_chart(
            fig_scatter,
            use_container_width=True
        )

    # 3D visualization using three consecutive years.
    if {
        "2021 Accidents",
        "2022 Accidents",
        "2023 Accidents"
    }.issubset(accidents.columns):

        three_d = accidents[[
            "State",
            "2021 Accidents",
            "2022 Accidents",
            "2023 Accidents"
        ]].copy()

        for col in [
            "2021 Accidents",
            "2022 Accidents",
            "2023 Accidents"
        ]:
            three_d[col] = pd.to_numeric(
                three_d[col], errors="coerce"
            )

        three_d = three_d.dropna()

        fig_3d = px.scatter_3d(
            three_d,
            x="2021 Accidents",
            y="2022 Accidents",
            z="2023 Accidents",
            hover_name="State",
            title="3D Accident Pattern Across 2021–2023"
        )
        st.plotly_chart(
            fig_3d,
            use_container_width=True
        )

    # --------------------------------------------------------
    # DATA STORYTELLING
    # --------------------------------------------------------

    st.markdown("### 3. Data Storytelling")

    st.markdown(
        "**What happened?** Historical accident counts vary substantially "
        "across states and years. The dashboard visualizations show the "
        "direction and magnitude of these changes."
    )

    st.markdown(
        "**Why did it happen?** The predictive framework evaluates accident, "
        "transport and environmental information together. SHAP and LIME "
        "identify which model inputs influenced individual predictions."
    )

    st.markdown(
        "**What is likely to happen?** The trained classification models "
        "estimate the historical accident-risk trend category under the "
        "defined evaluation protocol."
    )

    st.markdown(
        "**What should we do?** Use the predicted risk signal together with "
        "current weather assessment and model explanations to prioritize "
        "monitoring and safety precautions. These are decision-support "
        "signals, not guaranteed forecasts or causal recommendations."
    )

    # --------------------------------------------------------
    # DECISION SUPPORT
    # --------------------------------------------------------

    st.markdown("### 4. Decision Support")

    decision_state = st.selectbox(
        "Select region for decision support",
        states,
        key="decision_support_state"
    )

    decision_risk, decision_confidence, _ = predict_risk(
        decision_state
    )

    d1, d2 = st.columns(2)
    d1.metric("Historical Risk Signal", decision_risk)
    d2.metric("Model Confidence", f"{decision_confidence:.0%}")

    if decision_risk == "High":
        st.error(
            "Priority action: increase monitoring, review current weather "
            "conditions and apply additional road-safety precautions."
        )
    elif decision_risk == "Medium":
        st.warning(
            "Priority action: maintain enhanced monitoring and review "
            "current environmental conditions before travel decisions."
        )
    else:
        st.success(
            "Priority action: continue routine monitoring while checking "
            "current conditions before operational decisions."
        )

    # --------------------------------------------------------
    # RESPONSIBLE AI
    # --------------------------------------------------------

    st.markdown("### 5. Responsible AI & Limitations")

    st.info(
        "This project is an analytical prototype. Predictions represent "
        "patterns learned from the available datasets and should not be "
        "treated as causal conclusions or guaranteed safety outcomes."
    )

    responsible_points = [
        "The accident data is aggregated at state/year level, so individual-road conclusions are not supported.",
        "Transport observations are static and do not represent live passenger load or real-time traffic density.",
        "Weather data has different spatial and temporal granularity from the accident data and requires harmonization.",
        "SHAP and LIME explain model behavior; they do not prove that a feature causes accidents.",
        "Small and uneven group sizes can affect generalization and model stability.",
        "Sensitive personal information is not used in the current datasets.",
        "Future deployment should include monitoring for data drift, fairness, uncertainty and model performance over time."
    ]

    for point in responsible_points:
        st.write("•", point)


# ============================================================
# 8. MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.subheader(
        "📈 Final ML + DL Model Performance"
    )

    st.caption(
        "Final strict multimodal experiment using 5-fold GroupKFold "
        "by State, 21 independent features and 143 samples."
    )

    # ========================================================
    # LOAD FINAL STRICT EXPERIMENT RESULTS
    # ========================================================

    strict_path = RESULTS_DIR / "strict_ml_dl_comparison.csv"
    selection_path = RESULTS_DIR / "strict_ml_dl_selection.csv"

    try:

        strict_df = pd.read_csv(strict_path)

    except Exception:

        st.error(
            "Final strict ML + DL results are unavailable. "
            "Run src/hybrid_optimizer.py first."
        )
        st.stop()

    strict_df.columns = [
        column.strip()
        for column in strict_df.columns
    ]

    # --------------------------------------------------------
    # Normalize result columns
    # --------------------------------------------------------

    required_columns = [
        "Model",
        "Type",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "Time",
        "Fold F1 Mean",
        "Fold F1 Std"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in strict_df.columns
    ]

    if missing_columns:

        st.error(
            "The final result file is missing columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    # Convert metrics safely to numeric
    for column in required_columns[2:]:

        strict_df[column] = pd.to_numeric(
            strict_df[column],
            errors="coerce"
        )

    # Sort by final weighted F1
    strict_df = strict_df.sort_values(
        by="F1",
        ascending=False
    ).reset_index(drop=True)

    # ========================================================
    # FINAL MODEL SELECTION
    # ========================================================

    baseline_df = strict_df[
        strict_df["Type"].isin(
            [
                "Machine Learning",
                "Deep Learning"
            ]
        )
    ]

    hybrid_df = strict_df[
        strict_df["Type"] == "ML + DL Hybrid"
    ]

    best_baseline = baseline_df.loc[
        baseline_df["F1"].idxmax()
    ]

    best_hybrid = hybrid_df.loc[
        hybrid_df["F1"].idxmax()
    ]

    improvement = (
        float(best_hybrid["F1"])
        - float(best_baseline["F1"])
    )

    relative_improvement = (
        improvement
        / float(best_baseline["F1"])
    )

    # ========================================================
    # TOP METRIC CARDS
    # ========================================================

    st.markdown(
        "### 🏆 Final Model Selection"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Best Baseline",
        best_baseline["Model"],
        f"F1 {best_baseline['F1']:.4f}"
    )

    c2.metric(
        "Best Hybrid",
        best_hybrid["Model"],
        f"F1 {best_hybrid['F1']:.4f}"
    )

    c3.metric(
        "F1 Improvement",
        f"+{improvement:.4f}",
        f"+{relative_improvement:.2%}"
    )

    c4.metric(
        "Evaluation",
        "5-fold GroupKFold",
        "Grouped by State"
    )

    # ========================================================
    # FINAL RESULTS TABLE
    # ========================================================

    st.markdown(
        "### 📋 Final ML + DL Comparison"
    )

    display_df = strict_df[
        [
            "Model",
            "Type",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "Time"
        ]
    ].copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]:

        display_df[column] = (
            display_df[column] * 100
        ).round(2)

    display_df["Time"] = display_df["Time"].round(3)

    display_df = display_df.rename(
        columns={
            "Accuracy": "Accuracy (%)",
            "Precision": "Precision (%)",
            "Recall": "Recall (%)",
            "F1": "F1 (%)",
            "ROC-AUC": "ROC-AUC (%)",
            "Time": "Execution Time (s)"
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # CONFUSION MATRIX — CLASSIFICATION ERRORS
    # ========================================================

    st.markdown("### Confusion Matrix — Accident Trend Classification")
    st.caption(
        "Rows show the actual class; columns show the predicted class. "
        "Off-diagonal cells represent classification errors."
    )

    # Use held-out / out-of-fold predictions only, not training predictions.
    predictions_path = RESULTS_DIR / "strict_predictions.csv"
    confusion_path = RESULTS_DIR / "strict_confusion_matrix.csv"

    if predictions_path.exists():
        try:
            predictions_df = pd.read_csv(predictions_path)
            predictions_df.columns = [str(c).strip() for c in predictions_df.columns]
            column_map = {str(c).strip().lower(): c for c in predictions_df.columns}

            actual_col = next(
                (column_map[n] for n in
                 ("actual", "y_true", "true_label", "actual_class")
                 if n in column_map), None
            )
            predicted_col = next(
                (column_map[n] for n in
                 ("predicted", "y_pred", "predicted_label", "predicted_class", "prediction")
                 if n in column_map), None
            )
            model_col = next(
                (column_map[n] for n in ("model", "model_name") if n in column_map),
                None
            )

            if actual_col is None or predicted_col is None:
                st.warning(
                    "strict_predictions.csv must contain actual and predicted "
                    "class columns, for example y_true and y_pred."
                )
            else:
                if model_col is not None:
                    selected_predictions = predictions_df[
                        predictions_df[model_col].astype(str) == str(best_hybrid["Model"])
                    ]
                    if not selected_predictions.empty:
                        predictions_df = selected_predictions

                actual = predictions_df[actual_col].astype(str).str.strip()
                predicted = predictions_df[predicted_col].astype(str).str.strip()
                valid = actual.ne("") & predicted.ne("")
                actual, predicted = actual[valid], predicted[valid]

                preferred = ["Decreasing", "Stable", "Increasing"]
                observed = list(dict.fromkeys(list(actual.unique()) + list(predicted.unique())))
                labels = [x for x in preferred if x in observed]
                labels += [x for x in observed if x not in labels]

                if len(actual) == 0:
                    st.warning("No valid held-out predictions were found.")
                else:
                    from sklearn.metrics import confusion_matrix

                    matrix = confusion_matrix(actual, predicted, labels=labels)
                    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
                    fig_cm = px.imshow(
                        matrix_df,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="Blues",
                        labels={
                            "x": "Predicted Class",
                            "y": "Actual Class",
                            "color": "Samples"
                        },
                        title=f"Confusion Matrix — {best_hybrid['Model']}"
                    )
                    fig_cm.update_layout(
                        template="plotly_dark",
                        height=480,
                        xaxis_side="bottom"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

        except Exception as error:
            st.error("The confusion matrix could not be generated.")
            with st.expander("View confusion-matrix error"):
                st.code(str(error))

    elif confusion_path.exists():
        try:
            matrix_df = pd.read_csv(confusion_path, index_col=0)
            matrix_df = matrix_df.apply(pd.to_numeric, errors="coerce")
            matrix_df = matrix_df.dropna(axis=0, how="all").dropna(axis=1, how="all")
            fig_cm = px.imshow(
                matrix_df,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                labels={
                    "x": "Predicted Class",
                    "y": "Actual Class",
                    "color": "Samples"
                },
                title="Confusion Matrix — Accident Trend Classification"
            )
            fig_cm.update_layout(template="plotly_dark", height=480, xaxis_side="bottom")
            st.plotly_chart(fig_cm, use_container_width=True)
        except Exception as error:
            st.error("strict_confusion_matrix.csv could not be read.")
            with st.expander("View confusion-matrix error"):
                st.code(str(error))
    else:
        st.info(
            "To display a valid confusion matrix, save held-out / out-of-fold "
            "predictions as results/strict_predictions.csv with columns such as "
            "y_true and y_pred (plus an optional Model column). The app does not "
            "invent counts or evaluate predictions on training data."
        )

    # ========================================================
    # GRAPH 1: PRECISION / RECALL / F1
    # ========================================================

    st.markdown(
        "### 📊 Precision, Recall and F1-Score"
    )

    metric_plot = strict_df[
        [
            "Model",
            "Precision",
            "Recall",
            "F1"
        ]
    ].copy()

    metric_plot = metric_plot.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    metric_plot["Score"] = (
        metric_plot["Score"] * 100
    )

    fig_metrics = px.bar(
        metric_plot,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text="Score",
        title=(
            "Comparative Model Performance on "
            "Smart Transportation Risk-Trend Prediction"
        )
    )

    fig_metrics.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        cliponaxis=False
    )

    fig_metrics.update_layout(
        template="plotly_dark",
        xaxis_title="Models / Approaches",
        yaxis_title="Score (%)",
        yaxis_range=[0, 90],
        legend_title="Metric",
        height=650,
        margin=dict(
            l=40,
            r=30,
            t=80,
            b=180
        ),
        xaxis=dict(
            tickangle=-28
        )
    )

    st.plotly_chart(
        fig_metrics,
        use_container_width=True
    )

    # ========================================================
    # GRAPH 2: F1 COMPARISON
    # ========================================================

    st.markdown(
        "### 📈 F1-Score Ranking"
    )

    f1_plot = strict_df[
        [
            "Model",
            "F1",
            "Type"
        ]
    ].copy()

    f1_plot["F1 (%)"] = (
        f1_plot["F1"] * 100
    )

    f1_plot = f1_plot.sort_values(
        by="F1 (%)",
        ascending=True
    )

    fig_f1 = px.bar(
        f1_plot,
        x="F1 (%)",
        y="Model",
        color="Type",
        orientation="h",
        text="F1 (%)",
        title="Final F1-Score Comparison"
    )

    fig_f1.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        cliponaxis=False
    )

    fig_f1.update_xaxes(
        title="Weighted F1-Score (%)",
        range=[0, 85]
    )

    fig_f1.update_layout(
        template="plotly_dark",
        height=560,
        margin=dict(
            l=30,
            r=100,
            t=70,
            b=50
        )
    )

    st.plotly_chart(
        fig_f1,
        use_container_width=True
    )

    # ========================================================
    # GRAPH 3: BEST HYBRID WEIGHTS
    # ========================================================

    st.markdown(
        "### ⚖️ Selected Hybrid — Average Learned Weights"
    )

    weight_df = pd.DataFrame(
        {
            "Model": [
                "Gradient Boosting",
                "XGBoost",
                "MLP/FNN"
            ],
            "Weight": [
                0.20,
                0.36,
                0.44
            ]
        }
    )

    weight_df["Weight (%)"] = (
        weight_df["Weight"] * 100
    )

    fig_weights = px.bar(
        weight_df,
        x="Model",
        y="Weight (%)",
        text="Weight (%)",
        title=(
            "Gradient Boosting + XGBoost + MLP/FNN "
            "Weighted Probability Fusion"
        )
    )

    fig_weights.update_traces(
        texttemplate="%{text:.0f}%",
        textposition="outside"
    )

    fig_weights.update_yaxes(
        title="Average Weight (%)",
        range=[0, 55]
    )

    fig_weights.update_layout(
        template="plotly_dark",
        height=420
    )

    st.plotly_chart(
        fig_weights,
        use_container_width=True
    )

    # ========================================================
    # CROSS-VALIDATION STABILITY
    # ========================================================

    st.markdown(
        "### 📐 Cross-Validation Stability"
    )

    cv_plot = strict_df[
        [
            "Model",
            "Fold F1 Mean",
            "Fold F1 Std"
        ]
    ].copy()

    cv_plot["Fold F1 Mean (%)"] = (
        cv_plot["Fold F1 Mean"] * 100
    )

    cv_plot["Fold F1 Std (%)"] = (
        cv_plot["Fold F1 Std"] * 100
    )

    fig_cv = px.bar(
        cv_plot,
        x="Model",
        y="Fold F1 Mean (%)",
        error_y="Fold F1 Std (%)",
        color="Fold F1 Mean (%)",
        text="Fold F1 Mean (%)",
        title="Mean Fold F1 with Standard Deviation"
    )

    fig_cv.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig_cv.update_yaxes(
        title="Mean Fold F1 (%)",
        range=[0, 90]
    )

    fig_cv.update_layout(
        template="plotly_dark",
        height=600,
        margin=dict(
            l=40,
            r=30,
            t=70,
            b=180
        ),
        xaxis=dict(
            tickangle=-28
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig_cv,
        use_container_width=True 
    )

    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    st.markdown(
        "### 💡 Final Performance Interpretation"
    )

    st.success(
        f"The proposed **{best_hybrid['Model']}** hybrid achieved "
        f"the highest weighted F1-score of "
        f"**{best_hybrid['F1']:.4f}**, compared with "
        f"**{best_baseline['F1']:.4f}** for the best baseline "
        f"Random Forest. This is an absolute improvement of "
        f"**{improvement:.4f} F1 ({relative_improvement:.2%} relative)**."
    )

    st.write(
        "The hybrid was selected based primarily on weighted F1-score. "
        "All reported metrics are retained so that improvements and "
        "trade-offs remain transparent."
    )

    st.warning(
        f"The selected hybrid ROC-AUC is "
        f"{best_hybrid['ROC-AUC']:.4f}, while the Random Forest baseline "
        f"has ROC-AUC {best_baseline['ROC-AUC']:.4f}. Therefore, the "
        "hybrid should not be described as superior on every metric."
    )

    st.info(
        "Evaluation protocol: 5-fold GroupKFold by State. "
        "The hybrid uses leakage-safe weighted probability fusion, "
        "with its weights optimized inside the training folds. "
        "The experiment uses 143 samples and 21 independent features."
    )
