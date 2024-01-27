import pandas as pd
from src.data.load_data import load_interim_reddit_data
from sklearn.model_selection import train_test_split

TARGET_VARIABLES = [
    "max_climbed_grade",
    "recent_max_climbed_grade",
    "consistent_grade",
    "inferred_max_grade",
    "inferred_recent_grade",
    "inferred_consistent_grade",
    "median_grade",
    "median_grade_inferred",
]

TRAINING_TYPE_VARIABLES = [
    "climbing_activities",
    "avg_climbing_hours_weekly",
    "avg_training_hours_weekly",
    "climbing_freq_weekly",
    "endurance_training_freq_weekly",
    "strength_training_freq_weekly",
    "strength_training_hours_weekly",
    "campus_board_freq_weekly",
    "campus_board_hours_weekly",
    "hangboard_style_max_weight",
    "hangboard_style_min_edge",
    "hangboard_style_no_hangs",
    "hangboard_style_one_arm_hang_program",
    "hangboard_style_other_protocol",
    "hangboard_style_repeaters",
    "grip_back_2",
    "grip_back_3",
    "grip_front_2",
    "grip_front_3",
    "grip_full_crimp",
    "grip_half_crimp",
    "grip_middle_2",
    "grip_monos",
    "grip_no_hangs",
    "grip_open_crimp",
    "grip_pinch",
    "grip_slopers",
    "endurance_",
    "endurance_4x4",
    "endurance_arc",
    "endurance_feet_on_campusing_",
    "endurance_hangboard_repeater_protocols",
    "endurance_i_don_t_train_for_endurance",
    "endurance_laps_of_routes",
    "endurance_max_moves",
    "endurance_other",
    "endurance_route_climbing_intervals",
    "endurance_systems_boards",
    "endurance_threshold_intervals",
    "strength_antagonists",
    "strength_core",
    "strength_legs",
    "strength_no_other_strength_training",
    "strength_upper_body_pulling",
    "strength_upper_body_pushing",
    "running",
    "yoga",
    "cardio",
    "stretching",
    "lifting",
]

REDUNDANT_VARIABLES = [
    "gender",
    "max_climbed_grade_v",
    "recent_max_climbed_grade_v",
    "consistent_grade_v",
    "max_route_grade_ewbank",
    "recent_max_route_grade_ewbank",
    "consistent_route_grade_ewbank",
    "max_climbed_grade_fb",
    "recent_max_climbed_grade_fb",
    "consistent_grade_fb",
    "max_route_grade_french",
    "inferred_max_grade_v",
    "recent_max_route_grade_french",
    "inferred_recent_grade_v",
    "consistent_route_grade_french",
    "inferred_consistent_grade_v",
    "weight_lbs",
    "timestamp",
    "strength_training_type",
    "other_activities",
    "hangboard_grips",
    "hangboard_style",
    "endurance_training_type",
    "max_pull_ups",
]

VARIABLES_TO_DROP = ["max_l_sit_sec", "ape_diff", "max_push_ups"]


def main():
    # Load interim Reddit data
    data = load_interim_reddit_data()

    data["median_grade"] = data[["max_climbed_grade", "recent_max_climbed_grade", "consistent_grade"]].median(axis=1)

    # Use infered grades as proxy
    data["median_grade_inferred"] = pd.concat(
        [
            data["max_climbed_grade"].fillna(data["inferred_max_grade"]),
            data["recent_max_climbed_grade"].fillna(data["inferred_recent_grade"]),
            data["consistent_grade"].fillna(data["inferred_consistent_grade"]),
        ],
        axis=1,
    ).median(axis=1)

    data = data.dropna(subset=["median_grade"])

    # Select the input data
    data_performance = (
        data.drop(TRAINING_TYPE_VARIABLES, axis=1)
        .drop(REDUNDANT_VARIABLES, axis=1)
        .drop(TARGET_VARIABLES, axis=1)
        .drop(VARIABLES_TO_DROP, axis=1)
    )
    data_training_style = data[TRAINING_TYPE_VARIABLES].copy()

    target_data = data[TARGET_VARIABLES].copy()

    # Split the data into training and testing sets
    (
        data_performance_train,
        data_performance_test,
        data_training_style_train,
        data_training_style_test,
        target_data_train,
        target_data_test,
    ) = train_test_split(data_performance, data_training_style, target_data, test_size=0.1, random_state=42)

    # Save the data
    data_performance_train.to_csv("data/processed/reddit_performance/X_train.csv", index=False)
    data_performance_test.to_csv("data/processed/reddit_performance/X_test.csv", index=False)

    data_training_style_train.to_csv("data/processed/reddit_training_style/X_train.csv", index=False)
    data_training_style_test.to_csv("data/processed/reddit_training_style/X_test.csv", index=False)

    target_data_train.to_csv("data/processed/y_train.csv", index=False)
    target_data_test.to_csv("data/processed/y_test.csv", index=False)


if __name__ == "__main__":
    main()
