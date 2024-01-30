import psycopg2
from psycopg2 import OperationalError
import logging as logger

logger.basicConfig(level=logger.INFO)


def connect_to_postgres(secrets: dict):
    try:
        connection = psycopg2.connect(
            user=secrets["user"],
            password=secrets["password"],
            host=secrets["host"],
            port=secrets["port"],
            database=secrets["database"],
        )
        cursor = connection.cursor()
        logger.info(connection.get_dsn_parameters(), "\n")
        cursor.execute("SELECT version();")
        record = cursor.fetchone()
        logger.info("You are connected to - ", record, "\n")

        return connection

    except (Exception, OperationalError) as error:
        logger.info("Error while connecting to PostgreSQL", error)


def store_in_db(conn: psycopg2.extensions.connection, data: dict):
    """Store feedback data in PostgreSQL database"""
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO climbharderapp (timestamp, age, height_cm, weight_kg, arm_span_cm, climbing_exp_years,"
        " hangboard_freq_weekly, hangboard_max_weight_half_crimp_kg, hangboard_max_weight_open_crimp_kg,"
        " hangboard_min_edge_half_crimp_mm, hangboard_min_edge_open_crimp_mm, max_weight_pull_ups_kg,"
        f" actual_bouldering_grade, actual_sport_grade) VALUES ('{data['timestamp']}', {data['age']},"
        f" {data['height_cm']}, {data['weight_kg']}, {data['arm_span_cm']}, {data['climbing_exp_years']},"
        f" {data['hangboard_freq_weekly']}, {data['hangboard_max_weight_half_crimp_kg']},"
        f" {data['hangboard_max_weight_open_crimp_kg']}, {data['hangboard_min_edge_half_crimp_mm']},"
        f" {data['hangboard_min_edge_open_crimp_mm']}, {data['max_weight_pull_ups_kg']},"
        f" '{data['actual_bouldering_grade']}', '{data['actual_sport_grade']}');"
    )
    conn.commit()
