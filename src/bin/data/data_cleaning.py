import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from typing import List

from utils.climbing_grades_mapping import (
    v_to_fb_mapping,
    v_to_numeric_mapping,
    french_sport_to_v_grade_mapping,
    ewbank_aus_to_french_grade_mapping,
)

LBS_TO_KG = 0.453592
KG_TO_LBS = 2.20462

UNSURE_VALUES = [
    "unsure",
    "unknown",
    "don't know",
    "idk",
    "Na",
    "Not sure",
    "Unsure",
    "-",
    "Can't do 18mm",
    "N/a",
    "not applicable ",
    "0; definitely don't know what this is either. This survey makes me feel like I should be training...",
]


def contains_numeric(value: str) -> bool:
    return any(char.isdigit() for char in value)


def contains_substring(input_string: str, substring_list: List[str]) -> bool:
    return any(substring in input_string for substring in substring_list)


def convert_arm_span_to_cm(value: str) -> float:
    """
    Convert a string to float value in centimeters.
    Handles cases like '1.5m', '1.5 m', '1.5m?',
    """
    if isinstance(value, str) and contains_numeric(value):
        val_num = float(re.sub(r"[^0-9.]", "", value))

        # values are likely in meters
        if val_num < 2:
            val_num *= 100

        # discard unrealistic values (single arm unlikely below 50 for adult)
        elif val_num > 250 or val_num == 0.0 or (val_num > 3 and val_num < 50):
            val_num = np.nan

        # single arm case likely
        elif val_num > 50 and val_num < 100:
            val_num = val_num * 2 + 40  # average torso lenght
        return val_num
    else:
        return np.nan


def convert_to_kg(value: str) -> float:
    """
    Convert a string to float value in kilograms.
    Handles cases like '10kg added weight', '18kg??', '+40kg', '90kg bodyweight + 35kg =125kg', etc.
    """

    if isinstance(value, str):
        # Specific cases (hardcoded, as it's not worth to handle generally)
        if "106kg total (36kg added to BW)" == value:
            return float(36)
        elif "195lbs total (including bodyweight) " in value:
            return 14.45044
        elif "90kg bodyweight + 35kg =125kg" in value:
            return 35
        else:
            if re.search(r"\?|unknown|unsure|don\'t know|na", value, re.I):
                return np.nan

            # Extract numeric value and unit from the string
            match = re.search(r"(\d+\.?\d*)\s*(kg|lbs?|kilo|pounds?)?", value, re.I)
            if match:
                num, unit = match.groups()
                num = float(num)
                if unit and unit.lower() in ["lbs", "lb", "pound", "pounds"]:
                    num *= LBS_TO_KG  # Convert pounds to kilograms
                return num

            # Try to convert to float, set to NaN if unsuccessful
            try:
                return float(value)
            except ValueError:
                return np.nan

    return np.nan


def extract_years_range_and_average(value: str) -> float:
    """
    Extract the average years of experience from a string.
    Handles cases like '5-6 years',
    """
    if "More than" in value:
        return 15  # Assuming more than 15 years is represented as 15
    elif value == ".5 - 1 years":
        return 0.75
    elif value == "0 - .5 years":
        return 0.25
    else:
        num = re.findall(r"\d+\.\d+|\d+", value)
        lower, upper = map(float, num)
        avg = (lower + upper) / 2
        return avg


def convert_weighted_hangboard_to_kg(value: str) -> float:
    """
    Convert a string to float value in kilograms.
    Handles cases like '10kg added weight', '18kg??', '+40kg', '90kg bodyweight + 35kg =125kg', etc.
    """

    if isinstance(value, str):
        # Check for uncertain or unknown values
        if any([nan_case == value for nan_case in UNSURE_VALUES]):
            return np.nan

        # Handle specific cases
        match = re.search(r"(-?\d+\.?\d*)kg added weight", value, re.I)
        if match:
            return float(match.group(1))

        match = re.search(r"(-?\d+\.?\d*)kg\?\?", value, re.I)
        if match:
            return float(match.group(1))

        # Extract numeric value and unit from the string
        match = re.search(r"(-?\d+\.?\d*)\s*(kg|lbs?|kilo|pounds?)?", value, re.I)
        if match:
            num, unit = match.groups()
            num = float(num)
            if unit and unit.lower() in ["lbs", "lb", "pound", "pounds"]:
                num *= LBS_TO_KG  # Convert pounds to kilograms
            return num

        # Try to convert to float, set to NaN if unsuccessful
        try:
            return float(value)
        except ValueError:
            return np.nan

    return np.nan


def convert_weighted_hangboard_to_mm(value: str) -> float:
    """
    Convert a string to float value in millimeters.
    """
    if isinstance(value, str):
        if any([nan_case == value for nan_case in UNSURE_VALUES]):
            # Set uncertain or unknown values to NaN
            return np.nan
        elif "mm" in value.lower():
            return float(re.sub(r"[^0-9.]", "", value.split("mm")[0]))
        else:
            # Try to convert to float, set to NaN if unsuccessful
            try:
                return float(value)
            except ValueError:
                return np.nan
    else:
        return np.nan


def convert_max_push_ups(value: str) -> float:
    """
    Convert a string to float number of push ups.
    """
    if isinstance(value, str) and contains_numeric(value):
        val_num = re.sub(r"[^0-9.]", "", value)
        if len(val_num) == 4:
            return (float(val_num[:2]) + float(val_num[2:])) / 2
        else:
            return val_num
    else:
        return np.nan


def convert_max_l_sit_sec(value: str) -> float:
    """
    Convert a string to float number of seconds.
    """
    if (
        isinstance(value, str)
        and contains_numeric(value)
        and not contains_substring(
            value,
            UNSURE_VALUES,
        )
    ):
        val_low = value.lower()
        if contains_substring(val_low, ["s", "sec", "seconds", "second"]):
            return float(re.sub(r"[^0-9.]", "", value))
        elif ":" in val_low:
            pattern = r"(\d+(:\d+)?(?:\.\d+)?)"
            val_num = re.findall(pattern, val_low)
            min, s = val_num[0][0].split(":")
            return float(min) * 60.0 + float(s)
        elif contains_substring(value.lower(), ["min", "minutes", "seconds", "second"]):
            return float(re.sub(r"[^0-9.]", "", value)) * 60.0
        else:
            return float(re.sub(r"[^0-9.]", "", value))
    else:
        return np.nan


def load_data() -> pd.DataFrame:
    """
    Load the data from the csv file.
    """
    # Set the project path to the current directory
    project_path = Path(os.getcwd())

    # Now you can use this path to read the file
    data_path = project_path / "data/raw/reddit_data.csv"

    data = pd.read_csv(data_path)

    data.columns = [
        "timestamp",
        "gender",
        "height_cm",
        "weight_kg",
        "arm_span_cm",
        "climbing_exp_years",
        "climbing_activities",
        "max_climbed_grade_v",
        "recent_max_climbed_grade_v",
        "consistent_grade_v",
        "max_route_grade_ewbank",
        "recent_max_route_grade_ewbank",
        "consistent_route_grade_ewbank",
        "climbing_freq_weekly",
        "avg_climbing_hours_weekly",
        "avg_training_hours_weekly",
        "hangboard_freq_weekly",
        "hangboard_grips",
        "hangboard_style",
        "hangboard_max_weight_half_crimp_kg",
        "hangboard_max_weight_open_crimp_kg",
        "hangboard_min_edge_half_crimp_mm",
        "hangboard_min_edge_open_crimp_mm",
        "campus_board_freq_weekly",
        "campus_board_hours_weekly",
        "endurance_training_freq_weekly",
        "endurance_training_type",
        "strength_training_freq_weekly",
        "strength_training_hours_weekly",
        "strength_training_type",
        "other_activities",
        "max_pull_ups",
        "max_weight_pull_ups_kg",
        "max_push_ups",
        "max_l_sit_sec",
    ]

    return data


def fix_and_clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fix and clean the data.
    """
    data = data.replace("I don't boulder", pd.NA).replace("I don't climb routes", pd.NA)
    data["climbing_activities"] = (
        data["climbing_activities"].apply(lambda x: x.lower()).astype("category")
    )

    data["weight_kg"] = data["weight_kg"].replace(
        {
            "135 pounds....so....65 kg?": 61.235,
            "51-53...": 52,
            "72 kg": 72,
            "~55": 55,
            "78kg": 78,
            "82,5": 82.5,
            "70 kg ": 70,
            "66kg": 66,
        }
    )

    # someone did not give a damn and filled the form with random numbers
    data.drop(data[data["weight_kg"] == "889"].index, inplace=True)
    data["weight_kg"] = data["weight_kg"].astype(float)

    # weight is likely in lbs in those cases
    data.loc[data.query("weight_kg > 120").index, ["weight_kg"]] = (
        data.query("weight_kg > 120")["weight_kg"] * LBS_TO_KG
    )

    # someone accidentaly switched weight with height
    data.loc[data["height_cm"] == "62", ["height_cm", "weight_kg"]] = [172.0, 62]

    data["height_cm"] = data.height_cm.replace(
        {
            "173 cm": 173,
            "5 ft 8inches. Im amurican i dont know what centimeters are": 172.72,
            "167cm": 163,
            "1.67": 167,
            "1.75": 175,
            "1.68": 168,
            "1295": np.nan,
            "8": np.nan,
            "110": np.nan,
        }
    )
    # someone did not give a damn and filled the form with random numbers
    data.drop(data[data["height_cm"] == "8"].index, inplace=True)
    data.drop(data[data["height_cm"] == "110"].index, inplace=True)
    data["height_cm"] = data["height_cm"].astype(float)

    data["arm_span_cm"] = data["arm_span_cm"].replace(
        {"5 ft 10 inches": 177.8, "0": np.nan}
    )
    data["max_pull_ups"] = data["max_pull_ups"].replace(UNSURE_VALUES, np.nan)
    return data


def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the data.
    """

    data["climbing_exp_years"] = (
        data["climbing_exp_years"].apply(extract_years_range_and_average).astype(float)
    )
    data["arm_span_cm"] = data["arm_span_cm"].apply(convert_arm_span_to_cm)

    # Add some derived columns
    data["weight_lbs"] = data["weight_kg"] * 2.20462
    data["bmi"] = data["weight_kg"] / ((data["height_cm"] / 100) ** 2)
    data["ape_index"] = data["arm_span_cm"] / data["height_cm"]
    data["ape_diff"] = data["arm_span_cm"] - data["height_cm"]

    # Convert the bouldering grades
    data["max_climbed_grade_fb"] = data["max_climbed_grade_v"].map(v_to_fb_mapping)
    data["recent_max_climbed_grade_fb"] = data["recent_max_climbed_grade_v"].map(
        v_to_fb_mapping
    )
    data["consistent_grade_fb"] = data["consistent_grade_v"].map(v_to_fb_mapping)

    data["max_climbed_grade"] = (
        data["max_climbed_grade_v"].map(v_to_numeric_mapping).astype(float)
    )
    data["recent_max_climbed_grade"] = (
        data["recent_max_climbed_grade_v"].map(v_to_numeric_mapping).astype(float)
    )
    data["consistent_grade"] = (
        data["consistent_grade_v"].map(v_to_numeric_mapping).astype(float)
    )

    # Convert the route grades
    data["max_route_grade_french"] = data["max_route_grade_ewbank"].map(
        ewbank_aus_to_french_grade_mapping
    )
    data["inferred_max_grade_v"] = data["max_route_grade_french"].map(
        french_sport_to_v_grade_mapping
    )
    data["inferred_max_grade"] = data["inferred_max_grade_v"].map(v_to_numeric_mapping)

    data["recent_max_route_grade_french"] = data["recent_max_route_grade_ewbank"].map(
        ewbank_aus_to_french_grade_mapping
    )
    data["inferred_recent_grade_v"] = data["recent_max_route_grade_french"].map(
        french_sport_to_v_grade_mapping
    )
    data["inferred_recent_grade"] = data["inferred_recent_grade_v"].map(
        v_to_numeric_mapping
    )
    data["consistent_route_grade_french"] = data["consistent_route_grade_ewbank"].map(
        ewbank_aus_to_french_grade_mapping
    )
    data["inferred_consistent_grade_v"] = data["consistent_route_grade_french"].map(
        french_sport_to_v_grade_mapping
    )
    data["inferred_consistent_grade"] = data["inferred_consistent_grade_v"].map(
        v_to_numeric_mapping
    )

    data["hangboard_max_weight_half_crimp_kg"] = (
        data["hangboard_max_weight_half_crimp_kg"]
        .apply(convert_weighted_hangboard_to_kg)
        .astype(float)
    )
    data["hangboard_max_weight_open_crimp_kg"] = (
        data["hangboard_max_weight_open_crimp_kg"]
        .apply(convert_weighted_hangboard_to_kg)
        .astype(float)
    )
    data["hangboard_min_edge_half_crimp_mm"] = (
        data["hangboard_min_edge_half_crimp_mm"]
        .apply(convert_weighted_hangboard_to_mm)
        .apply(lambda x: np.nan if x > 30 else x)
    )
    data["hangboard_min_edge_open_crimp_mm"] = (
        data["hangboard_min_edge_open_crimp_mm"]
        .apply(convert_weighted_hangboard_to_mm)
        .apply(lambda x: np.nan if x > 30 else x)
    )

    data["max_weight_pull_ups_kg"] = (
        data["max_weight_pull_ups_kg"].apply(convert_to_kg).astype(float)
    )
    # manually fix outliers (as it seems that the values are in lbs + bw)
    data.loc[[60, 367, 419, 558, 576], "max_weight_pull_ups_kg"] = (
        data.loc[[60, 367, 419, 558, 576], "max_weight_pull_ups_kg"]
        - data.loc[[60, 367, 419, 558, 576], "weight_lbs"]
    ) * LBS_TO_KG

    data["max_pull_ups"] = pd.to_numeric(data["max_pull_ups"], errors="coerce")

    data["max_push_ups"] = (
        data["max_push_ups"].apply(convert_max_push_ups).astype(float)
    )

    data["max_l_sit_sec"] = (
        data["max_l_sit_sec"].apply(convert_max_l_sit_sec).astype(float)
    )

    # Add dummy columns
    data["endurance_training_type"] = (
        data["endurance_training_type"].fillna("").str.split(", ")
    )
    endurance_dummies = pd.get_dummies(
        data["endurance_training_type"].explode(), prefix="endurance"
    )
    endurance_dummies = endurance_dummies.groupby(endurance_dummies.index).agg(
        lambda x: x.max()
    )

    # Grip style dummies
    grip_dummies = (
        data["hangboard_grips"]
        .replace({"I don't Hangboard": "no_hangs"})
        .replace({" I don't Hangboard": "no_hangs"})
        .str.get_dummies(sep=", ")
    )

    # Rename the columns for clarity
    grip_dummies.columns = [
        "grip_" + col.replace(" ", "_") for col in grip_dummies.columns
    ]
    grip_dummies["grip_no_hangs"] = (
        grip_dummies["grip_I_don't_Hangboard"] + grip_dummies["grip_no_hangs"]
    )
    grip_dummies.drop(columns=["grip_I_don't_Hangboard"], inplace=True)
    grip_dummies = grip_dummies.astype(bool)

    # hangborad style dummies
    style_dummies = (
        data["hangboard_style"]
        .replace({"I don't hangboard": "no_hangs"})
        .str.get_dummies(sep=", ")
    )

    # Rename the columns for clarity
    style_dummies.columns = [
        "hangboard_style_" + col.replace(" ", "_") for col in style_dummies.columns
    ]
    style_dummies["hangboard_style_no_hangs"] = (
        style_dummies['hangboard_style_"no_hangs"']
        + style_dummies["hangboard_style_no_hangs"]
    )
    style_dummies.drop(columns=['hangboard_style_"no_hangs"'], inplace=True)
    style_dummies = style_dummies.astype(bool)

    data["strength_training_type"] = (
        data["strength_training_type"].fillna("").str.split(", ")
    )
    strength_dummies = pd.get_dummies(
        data["strength_training_type"].explode(), prefix="strength"
    )
    strength_dummies = strength_dummies.groupby(strength_dummies.index).agg(
        lambda x: x.max()
    )

    data = pd.concat(
        [data, style_dummies, grip_dummies, endurance_dummies, strength_dummies], axis=1
    )

    data["other_activities"] = data["other_activities"].fillna("").astype("category")

    # Define the activities to track
    activities_to_track = ["running", "yoga", "cardio", "stretching", "lifting"]

    # Create binary columns for each activity
    for activity in activities_to_track:
        data[activity] = data["other_activities"].str.contains(activity, case=False)

    return data


def main():
    data = load_data()
    data = fix_and_clean_data(data)
    data = transform_data(data)

    data.to_csv("data/interim/reddit_data.csv", index=False)

    return data


if __name__ == "__main__":
    main()
