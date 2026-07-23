from fastapi import FastAPI, UploadFile, File
import shutil
from pathlib import Path

from backend.app.predictor import predict_stress
from backend.app.ecg_preprocessing import (
    prepare_features_from_csv
)

app = FastAPI()

@app.get("/")
def home():

    return {
        "message": "ECG Stress Detection API Running"
    }

@app.post("/predict")
async def predict(
    baseline_file: UploadFile = File(...),
    target_file: UploadFile = File(...)
):

    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    baseline_path = temp_dir / baseline_file.filename
    target_path = temp_dir / target_file.filename

    with open(baseline_path, "wb") as buffer:
        shutil.copyfileobj(
            baseline_file.file,
            buffer
        )

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(
            target_file.file,
            buffer
        )

    features = prepare_features_from_csv(
        str(target_path),
        str(baseline_path),
        column_name="ecg"
    )

    result = predict_stress(features)
    baseline_path.unlink(missing_ok=True)
    target_path.unlink(missing_ok=True)

    return result
