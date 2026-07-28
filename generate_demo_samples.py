import os
import pickle
import numpy as np
import pandas as pd

# ----------------------------
# Configuration
# ----------------------------

SUBJECT_PATH = "WESAD/S2/S2.pkl"
OUTPUT_DIR = "samples"

LABELS = {
    1: "baseline",
    2: "high_stress",
    3: "medium_stress",
    4: "low_stress"
}

ECG_FS = 700
WINDOW_SECONDS = 20
WINDOW_SAMPLES = ECG_FS * WINDOW_SECONDS


# ----------------------------
# Helper Functions
# ----------------------------

def load_subject(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def extract_baseline(df):
    labels = df["label"]

    baseline_idx = np.where(labels == 1)[0]

    ecg = np.array(
        df["signal"]["chest"]["ECG"][
            baseline_idx[0]:baseline_idx[-1] + 1
        ][:, 0]
    )

    return ecg


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


def save_csv(signal, path):
    pd.DataFrame(signal, columns=["ECG"]).to_csv(path, index=False)


# ----------------------------
# Main
# ----------------------------

def main():

    subject = load_subject(SUBJECT_PATH)

    baseline = extract_baseline(subject)

    print(f"Baseline samples : {len(baseline)}")

    for label in [2, 3, 4]:

        target = find_first_constant_window(subject, label)

        if target is None:
            print(f"Could not find label {label}")
            continue

        folder = os.path.join(OUTPUT_DIR, LABELS[label])
        os.makedirs(folder, exist_ok=True)

        save_csv(
            baseline,
            os.path.join(folder, "baseline.csv")
        )

        save_csv(
            target,
            os.path.join(folder, "target.csv")
        )

        print(f"Created {folder}")

    print("\nDone.")


if __name__ == "__main__":
    main()