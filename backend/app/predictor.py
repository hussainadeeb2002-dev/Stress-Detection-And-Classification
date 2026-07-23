import torch
import numpy as np

from backend.app.model_loader import model

THRESHOLD = 0.5


def get_stress_level(score):

    if score <= 0.20:
        return "Very Low Stress"

    elif score <= 0.40:
        return "Mild Stress"

    elif score <= 0.60:
        return "Moderate Stress"

    elif score <= 0.80:
        return "High Stress"

    else:
        return "Severe Stress"


def get_interpretation(score):

    if score <= 0.20:
        return "Physiological signals appear calm and stable."

    elif score <= 0.40:
        return "Minor stress indicators detected."

    elif score <= 0.60:
        return "Moderate stress activity observed."

    elif score <= 0.80:
        return "Elevated stress response detected."

    else:
        return "Strong physiological stress pattern detected."


def predict_stress(features):

    features = np.array(features)

    x = torch.tensor(features).float().unsqueeze(0)

    with torch.no_grad():

        output = model(x).item()

    prediction = "Stress" if output > THRESHOLD else "No Stress"

    stress_level = get_stress_level(output)

    interpretation = get_interpretation(output)

    return {
        "prediction": prediction,
        "stress_score": float(output),
        "stress_level": stress_level,
        "interpretation": interpretation
    }