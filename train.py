# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from ultralytics import YOLO


def train_model():
    model = YOLO("yolo11n-obb.pt")
    model.train(
        data="mathon/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        patience=15,
        amp=False,
        verbose=True,
        project="runs/obb",
        name="yolo11n_mathon",
    )


if __name__ == "__main__":
    train_model()
