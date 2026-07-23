import torch
from pathlib import Path
from backend.app.model_definition import ClassifierECG

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "models" / "stress_detector.pth"

device = torch.device("cpu")

model = ClassifierECG(ngpu=0)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()