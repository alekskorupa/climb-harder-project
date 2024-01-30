# Climb-harder Project :climbing:

Streamlit web app designed to assist climbers in understanding their weaknesses and enhancing their performance. The application predicts climbing grades based on users' physical attributes and training habits, leveraging data obtained from a [reddit thread](https://www.reddit.com/r/climbharder/comments/gi7v2k/rquest_climbing_specific_datasets/). This project was developed for personal entertainment and skill enhancement and there are no serious plans to develop it further.

## Features

- **Grade estimation**: Users can input their physical attributes and training habits, and the application will estimate their climbing grades for both bouldering and sport climbing. The estimation is done by sending a POST request to a FastAPI application serving the scikit-learn model.

- **Feedback submission**: Users can provide their actual climbing grades for bouldering and sport climbing. This feedback, along with the user's physical attributes and training habits, is stored in a PostgreSQL database for future model improvement.

- **Comparison with other climbers**: Users can compare their physical attributes with other climbers who have the same bouldering grade. The comparison is visualized with box plots, where the user's values are indicated with a green line. 

## Demo

The application uses the Streamlit library for the web interface and can be deployed on Streamlit cloud, however to avoid costs of hosting model the model and database on the cloud it is currently not available. 

![Demo Image](images/interface_demo.png)

## Tech Stack

This project is built with the following technologies:

- **Programming Language**: Python
- **MLOps**: DVC, MLflow
- **Back-end**: FastAPI, Postgres 
- **Frontend**: Streamlit
- **Deployment**: Streamlit cloud

## Installation

TBC
