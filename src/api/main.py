import uvicorn

from fastapi import FastAPI
from fastapi import Request
import joblib
import numpy as np
from pathlib import Path
import pandas as pd
import logging

app = FastAPI(title="Climb Harder API", version="0.1.0", debug=True)

# Logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = Path(__file__).parents[2] / "data"
MODEL_PATH = DATA_PATH / "models/performance_model"

# Load the model
model = joblib.load(MODEL_PATH / "model.pkl")


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    logger.debug(f"Predicting for {data}")
    data = pd.DataFrame(data)
    prediction = int(model.predict(data))
    logger.debug(f"Prediction: {prediction}")
    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="debug", proxy_headers=True, reload=True)
