import streamlit as st
from src.utils.climbing_grades_mapping import (
    v_to_fb_mapping,
    numeric_to_v_mapping,
    v_grade_to_french_sport_mapping,
    fb_to_v_mapping,
    v_to_numeric_mapping,
    french_sport_to_v_grade_mapping,
)
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging as logger
import requests
from pathlib import Path

logger.basicConfig(level=logger.INFO)

APP_ENV = os.getenv("APP_ENV", "local")

if APP_ENV == "local":
    # Use local data and model
    DATA_PATH = Path(__file__).parents[2] / "data"
    REDDIT_DATA_PATH = DATA_PATH / "interim/reddit_data.csv"
    MODEL_URL = "http://localhost:8000/predict"
    reddit_data = pd.read_csv(REDDIT_DATA_PATH)
    try:
        reddit_data = pd.read_csv(REDDIT_DATA_PATH)
    except FileNotFoundError:
        logger.error(f"File not found: {REDDIT_DATA_PATH}")
        st.error("File not found. Please run `dvc pull` to make data available locally.")
else:
    # Connect to database
    from src.app.db_utils import connect_to_postgres, store_in_db

    secrets = st.secrets["postgres"]
    conn = connect_to_postgres(secrets=secrets)
    MODEL_URL = "https://climb-harder.herokuapp.com/predict"
    reddit_data = pd.read_sql_table("reddit_data", conn)

value_presets = {
    "age": {"display_name": "Age", "min": 10, "max": 100, "default": 30},
    "climbing_exp_years": {"display_name": "Climbing experience (years)", "min": 0, "max": 50, "default": 5},
    "height_cm": {"display_name": "Height (cm)", "min": 100, "max": 250, "default": 182},
    "weight_kg": {"display_name": "Weight (kg)", "min": 30.0, "max": 200.0, "default": 78.0},
    "arm_span_cm": {"display_name": "Arm span (cm)", "min": 100, "max": 250, "default": 188},
    "hangboard_freq_weekly": {"display_name": "Hangboard frequency (weekly)", "min": 0, "max": 7, "default": 1},
    "hangboard_max_weight_half_crimp_kg": {
        "display_name": "Hangboard max weight (kg) half crimp/18 mm/10s",
        "min": 0,
        "max": 200,
        "default": 55,
    },
    "hangboard_max_weight_open_crimp_kg": {
        "display_name": "Hangboard nax weight (kg) open crimp/18 mm/10s",
        "min": 0,
        "max": 200,
        "default": 55,
    },
    "hangboard_min_edge_half_crimp_mm": {
        "display_name": "Hangboard min edge (mm) half crimp/10s",
        "min": 0,
        "max": 50,
        "default": 8,
    },
    "hangboard_min_edge_open_crimp_mm": {
        "display_name": "Hangboard min edge (mm) open crimp/10s",
        "min": 0,
        "max": 50,
        "default": 7,
    },
    "max_weight_pull_ups_kg": {"display_name": "Max weight 5 pull ups (kg)", "min": 0, "max": 200, "default": 38},
    "bmi": {"display_name": "BMI", "min": 15, "max": 35, "default": 0},
    "ape_index": {"display_name": "Ape index", "min": 0.9, "max": 1.1, "default": 1.0},
    "ape_diff": {"display_name": "Ape difference", "min": -20, "max": 20, "default": 0},
}


def create_input(name, input_type):
    if input_type == "number":
        return st.sidebar.number_input(
            value_presets[name]["display_name"],
            min_value=value_presets[name]["min"],
            value=value_presets[name]["default"],
        )
    elif input_type == "slider":
        return st.sidebar.slider(
            value_presets[name]["display_name"],
            min_value=value_presets[name]["min"],
            max_value=value_presets[name]["max"],
            value=value_presets[name]["default"],
        )


def create_sidebar():
    st.sidebar.subheader("About you")
    inputs = {}
    for name in ["age", "height_cm", "weight_kg", "arm_span_cm", "climbing_exp_years"]:
        inputs[name] = create_input(name, input_type="number")

    inputs["gender"] = st.sidebar.selectbox("Gender", ("male", "female"))

    st.sidebar.subheader("Strenght training parameters")
    for name in [
        "hangboard_freq_weekly",
        "hangboard_max_weight_half_crimp_kg",
        "hangboard_max_weight_open_crimp_kg",
        "hangboard_min_edge_half_crimp_mm",
        "hangboard_min_edge_open_crimp_mm",
        "max_weight_pull_ups_kg",
    ]:
        inputs[name] = create_input(name, input_type="slider")

    # Compute additional features
    inputs["bmi"] = inputs["weight_kg"] / (inputs["height_cm"] / 100) ** 2
    inputs["ape_index"] = inputs["arm_span_cm"] / inputs["height_cm"]
    inputs["ape_diff"] = inputs["arm_span_cm"] - inputs["height_cm"]
    inputs["is_male"] = True if inputs["gender"] == "male" else False

    return inputs


def estimate_grades(inputs):
    logger.info("Estimating climbing grades")
    data = pd.Series(data=inputs).drop(["gender", "ape_diff", "age"]).to_frame().transpose()
    data = data.to_dict(orient="records")
    logger.info(f"Sending data to API: {data}")

    try:
        response = requests.post(MODEL_URL, json=data)
        response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx

        # Get the prediction from the response
        prediction = int(response.json()["prediction"])

        prediction_v = numeric_to_v_mapping[prediction]
        prediction_fb = v_to_fb_mapping[prediction_v]
        prediction_french_sport = v_grade_to_french_sport_mapping[prediction_v]
        st.write("Your estimated climbing grades are: ")
        st.write(f"Bouldering: {prediction_v} / ({prediction_fb})")
        st.write(f"Sport: {prediction_french_sport}")
        logger.info(
            f"Climbing grades estimated: {prediction_v} / ({prediction_fb} / {prediction_french_sport} (sport) "
        )
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        st.write(f"Request failed with status code {response.status_code}")
    except Exception as err:
        logger.error(f"Other error occurred: {err}")
        st.write("An error occurred.")


def submit_feedback(inputs, actual_bouldering_grade, actual_sport_grade):
    if APP_ENV == "local":
        st.write("Feedback submision is not available in the local environment")
    else:
        timestamp = pd.Timestamp.now()
        feedback_data = {
            "timestamp": str(timestamp),
            "age": inputs["age"],
            "height_cm": inputs["height_cm"],
            "weight_kg": inputs["weight_kg"],
            "arm_span_cm": inputs["arm_span_cm"],
            "climbing_exp_years": inputs["climbing_exp_years"],
            "hangboard_freq_weekly": inputs["hangboard_freq_weekly"],
            "hangboard_max_weight_half_crimp_kg": inputs["hangboard_max_weight_half_crimp_kg"],
            "hangboard_max_weight_open_crimp_kg": inputs["hangboard_max_weight_open_crimp_kg"],
            "hangboard_min_edge_half_crimp_mm": inputs["hangboard_min_edge_half_crimp_mm"],
            "hangboard_min_edge_open_crimp_mm": inputs["hangboard_min_edge_open_crimp_mm"],
            "max_weight_pull_ups_kg": inputs["max_weight_pull_ups_kg"],
            "actual_bouldering_grade": actual_bouldering_grade,
            "actual_sport_grade": actual_sport_grade,
        }
        logger.info(f"Feedback data: {feedback_data}")
        logger.info("Sending feedback to database")
        store_in_db(feedback_data)
        logger.info("Data inserted successfully")
        st.success("Thank you for your feedback! This will be used to improve the model and future predictions.")


def compare_grades(inputs, actual_bouldering_grade, reddit_data):
    recent_max_climbed_grade = v_to_numeric_mapping[fb_to_v_mapping[actual_bouldering_grade]]
    dist = reddit_data.query(f"recent_max_climbed_grade == {recent_max_climbed_grade}")

    cols = [
        "bmi",
        "height_cm",
        "ape_diff",
        "arm_span_cm",
        "climbing_exp_years",
        "hangboard_freq_weekly",
        "hangboard_max_weight_half_crimp_kg",
        "hangboard_max_weight_open_crimp_kg",
        "hangboard_min_edge_half_crimp_mm",
        "hangboard_min_edge_open_crimp_mm",
        "max_weight_pull_ups_kg",
    ]

    fig = make_subplots(rows=6, cols=2)

    for i, attribute_name in enumerate(cols):
        row = i // 2 + 1
        col = i % 2 + 1

        value = inputs[attribute_name]
        same_grade_data = dist[attribute_name]

        # Create box plot
        fig.add_trace(
            go.Box(
                y=same_grade_data, name=value_presets[attribute_name]["display_name"], marker_color="rgb(0, 0, 100)"
            ),
            row=row,
            col=col,
        )

        # Add vertical line for user's value
        fig.add_shape(type="line", x0=-0.4, x1=0.4, y0=value, y1=value, line=dict(color="Green"), row=row, col=col)

    fig.update_layout(height=1400, width=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


st.set_page_config(layout="wide")

st.title("Climber performance estimator", anchor="center")
st.write("This app is designed to assist climbers in understanding their weaknesses and enhancing their performance. ")

st.sidebar.header("Enter your climbing stats")
inputs = create_sidebar()

st.subheader("Grade estimation")
st.write(
    "Estimations are based on a linear regression model trained on data from data collected on Reddit. It estimates"
    " climbing grades based on your physical attributes and training habits. The model is trained on bouldering grades,"
    " but it can be used to estimate sport grades as well."
)
if st.button("Estimate"):
    estimate_grades(inputs)

st.subheader("Compare yourself to other climbers")
st.write(
    "Compare your physical attributes to other climbers with the same bouldering grade. Select the grade you have"
    ' climbed recently and press "Compare" button. Green line indicates your value. If plots are empty, there is no'
    " data available for this grade."
)
actual_bouldering_grade = st.selectbox(
    "Actual bouldering grade",
    (grade for grade in v_to_fb_mapping.values()),
)
actual_sport_grade = st.selectbox("Actual sport grade", (grade for grade in french_sport_to_v_grade_mapping.keys()))

if st.button("Compare"):
    compare_grades(inputs, actual_bouldering_grade, reddit_data)

st.subheader("Submit feedback")
st.write(
    "Submit your climbing grades to help improve the model. This data will be used to improve the model and future"
    " predictions."
)

if st.button("Submit"):
    if actual_bouldering_grade == "Select" or actual_sport_grade == "Select":
        st.warning("Please select your actual climbing grades before submitting feedback")
    else:
        submit_feedback(inputs, actual_bouldering_grade, actual_sport_grade)
