import pandas as pd
from pathlib import Path

# Get the root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_raw_reddit_data() -> pd.DataFrame:
    """
    Load the data from the csv file.
    """

    data_path = PROJECT_ROOT / "data/raw/reddit_data.csv"

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


def load_interim_reddit_data():
    data = pd.read_csv(PROJECT_ROOT / "data/interim/reddit_data.csv")
    return data


def load_reddit_performance_data():
    X_train = pd.read_csv(PROJECT_ROOT / "data/processed/reddit_performance/X_train.csv")
    X_test = pd.read_csv(PROJECT_ROOT / "data/processed/reddit_performance/X_test.csv")
    return X_train, X_test


def load_reddit_training_style_data():
    X_train = pd.read_csv(PROJECT_ROOT / "data/processed/reddit_training_style/X_train.csv")
    X_test = pd.read_csv(PROJECT_ROOT / "data/processed/reddit_training_style/X_test.csv")
    return X_train, X_test


def load_reddit_targets_data():
    y_train = pd.read_csv(PROJECT_ROOT / "data/processed/y_train.csv")
    y_test = pd.read_csv(PROJECT_ROOT / "data/processed/y_test.csv")
    return y_train, y_test
