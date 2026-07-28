import os
import pickle
import numpy as np
import pandas as pd

from backend.app.ecg_preprocessing import prepare_features_from_csv
from backend.app.predictor import predict_stress

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

SUBJECT_PATH = "WESAD/S2/S2.pkl"
OUTPUT_DIR = "samples"

LABELS = {
    2: "high_stress",
    3: "medium_stress",
    4: "low_stress",
}

ECG_FS = 700
WINDOW_SECONDS = 20
WINDOW_SAMPLES = ECG_FS * WINDOW_SECONDS

TEMP_DIR = "_temp_demo"
os.makedirs(TEMP_DIR, exist_ok=True)


# ----------------------------------------------------
# Helpers
# ----------------------------------------------------

def load_subject(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def extract_baseline(df):
    labels = df["label"]

    idx = np.where(labels == 1)[0]

    return np.array(
        df["signal"]["chest"]["ECG"][
            idx[0]: idx[-1] + 1
        ][:, 0]
    )


def save_csv(signal, path):
    pd.DataFrame(signal, columns=["ECG"]).to_csv(path, index=False)


def find_first_constant_window(df, desired_label):

    labels = df["label"]

    for start in range(
        0,
        len(labels) - WINDOW_SAMPLES,
        ECG_FS,
    ):

        if labels[start] != desired_label:
            continue

        window = labels[start:start + WINDOW_SAMPLES]

        if np.all(window == desired_label):

            ecg = np.array(
                df["signal"]["chest"]["ECG"][
                    start:start + WINDOW_SAMPLES
                ][:, 0]
            )

            return ecg

    return None


def find_best_medium_window(df, baseline):

    labels = df["label"]

    baseline_path = os.path.join(TEMP_DIR, "baseline.csv")
    target_path = os.path.join(TEMP_DIR, "target.csv")

    save_csv(baseline, baseline_path)

    best_signal = None
    best_score = None
    best_distance = float("inf")

    TARGET_SCORE = 0.45

    print("\nSearching for best medium-stress sample...\n")

    for start in range(
        0,
        len(labels) - WINDOW_SAMPLES,
        ECG_FS,
    ):

        if labels[start] != 3:
            continue

        window = labels[start:start + WINDOW_SAMPLES]

        if not np.all(window == 3):
            continue

        signal = np.array(
            df["signal"]["chest"]["ECG"][
                start:start + WINDOW_SAMPLES
            ][:, 0]
        )

        save_csv(signal, target_path)

        try:

            features = prepare_features_from_csv(
                target_path,
                baseline_path,
                column_name="ecg",
            )

            result = predict_stress(features)

            score = result["stress_score"]

            distance = abs(score - TARGET_SCORE)

            print(
                f"Window {start:7d} "
                f" Score = {score:.3f}"
            )

            if distance < best_distance:
                best_distance = distance
                best_score = score
                best_signal = signal.copy()

        except Exception:
            continue

    if best_signal is None:
        raise RuntimeError("No valid amusement window found.")

    print("\nBest medium window selected")
    print(f"Predicted score : {best_score:.3f}\n")

    return best_signal


# ----------------------------------------------------
# Main
# ----------------------------------------------------

def main():

    subject = load_subject(SUBJECT_PATH)

    baseline = extract_baseline(subject)

    print(f"Baseline samples : {len(baseline)}")

    for label in [2, 4]:

        target = find_first_constant_window(subject, label)

        folder = os.path.join(
            OUTPUT_DIR,
            LABELS[label],
        )

        os.makedirs(folder, exist_ok=True)

        save_csv(
            baseline,
            os.path.join(folder, "baseline.csv"),
        )

        save_csv(
            target,
            os.path.join(folder, "target.csv"),
        )

        print(f"Created {folder}")

    medium_target = find_best_medium_window(
        subject,
        baseline,
    )

    medium_folder = os.path.join(
        OUTPUT_DIR,
        "medium_stress",
    )

    os.makedirs(medium_folder, exist_ok=True)

    save_csv(
        baseline,
        os.path.join(
            medium_folder,
            "baseline.csv",
        ),
    )

    save_csv(
        medium_target,
        os.path.join(
            medium_folder,
            "target.csv",
        ),
    )

    print("Created samples/medium_stress")
    print("\nDone.")


if __name__ == "__main__":
    main()