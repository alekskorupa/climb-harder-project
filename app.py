import streamlit as st
import joblib
import numpy as np
from pathlib import Path
from src.utils.climbing_grades_mapping import (
    v_to_fb_mapping,
    numeric_to_v_mapping,
    v_grade_to_french_sport_mapping,
    fb_to_v_mapping,
    v_to_numeric_mapping,
)
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

model_path = Path(__file__).parent / "data/models/performance_model"

# Constants
MODEL_PATH = Path(__file__).parent / "data/models/performance_model"
REDDIT_DATA_PATH = "data/interim/reddit_data.csv"

# Load data and model
model = joblib.load(MODEL_PATH / "model.pkl")
reddit_data = pd.read_csv(REDDIT_DATA_PATH)


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

st.set_page_config(layout="wide")

st.title("Climber performance estimator", anchor="center")

col1, col2, col3 = st.columns([1, 1, 2])

col1.subheader("About you")

inputs = {}
for name in ["age", "height_cm", "weight_kg", "arm_span_cm", "climbing_exp_years"]:
    inputs[name] = col1.number_input(
        value_presets[name]["display_name"],
        min_value=value_presets[name]["min"],
        value=value_presets[name]["default"],
    )
inputs["gender"] = col1.selectbox("Gender", ("male", "female"))

col2.subheader("Grade estimation")
col2.write(
    "This app predicts climbing grades based on your physical attributes and training habits. Estimations are based on"
    " a linear regression model trained on data from data collected on Reddit. It estimates climbing grades based on"
    " your physical attributes and training habits. The model is trained on bouldering grades, but it can be used to"
    " estimate sport grades as well."
)
for name in [
    "hangboard_freq_weekly",
    "hangboard_max_weight_half_crimp_kg",
    "hangboard_max_weight_open_crimp_kg",
    "hangboard_min_edge_half_crimp_mm",
    "hangboard_min_edge_open_crimp_mm",
    "max_weight_pull_ups_kg",
]:
    inputs[name] = col2.slider(
        value_presets[name]["display_name"],
        min_value=value_presets[name]["min"],
        max_value=value_presets[name]["max"],
        value=value_presets[name]["default"],
    )


inputs["bmi"] = inputs["weight_kg"] / (inputs["height_cm"] / 100) ** 2
inputs["ape_index"] = inputs["arm_span_cm"] / inputs["height_cm"]
inputs["ape_diff"] = inputs["arm_span_cm"] - inputs["height_cm"]
inputs["is_male"] = True if inputs["gender"] == "male" else False


# When 'Predict' is clicked, make a prediction and display it
if col2.button("Estimate"):
    data = pd.Series(data=inputs).drop(["gender", "ape_diff", "age"]).to_frame().transpose()

    prediction = int(model.predict(data))
    prediction_v = numeric_to_v_mapping[prediction]
    prediction_fb = v_to_fb_mapping[prediction_v]
    prediction_french_sport = v_grade_to_french_sport_mapping[prediction_v]
    col2.write("Your estimated climbing grades are: ")
    col2.write(f"Bouldering: {prediction_v} / ({prediction_fb})")
    col2.write(f"Sport: {prediction_french_sport}")

col3.subheader("Compare yourself to other climbers")
col3.write(
    "Compare your physical attributes to other climbers with the same bouldering grade. Select the grade you have"
    ' climbed recently and press "Compare" button. Green line indicates your value. If plots are empty, there is no'
    " data available for this grade."
)

actual_bouldering_grade = col3.selectbox(
    "Bouldering grade",
    (
        "4",
        "5",
        "5+",
        "6A",
        "6B",
        "6C",
        "7A",
        "7A+",
        "7B",
        "7C",
        "7C+",
        "8A",
        "8A+",
        "8B",
        "8B+",
        "8C",
        "8C+",
        "9A",
    ),
)

if col3.button("Compare"):
    recent_max_climbed_grade = v_to_numeric_mapping[fb_to_v_mapping[actual_bouldering_grade]]
    dist = reddit_data.query(f"recent_max_climbed_grade == {recent_max_climbed_grade}")

    # Assuming inputs and dist are pandas DataFrame
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
    col3.plotly_chart(fig, use_container_width=True)
