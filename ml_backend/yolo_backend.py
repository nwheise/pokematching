import os
from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO

MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "outputs", "detection", "train", "weights", "best.pt"),
)

CLASSES = ["attached-energy", "attached-item", "card", "multicard"]


class YOLOBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.yolo = YOLO(MODEL_PATH)

    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            image_url = task["data"].get("image")
            image_path = self.get_local_path(image_url)

            results = self.yolo(image_path)[0]
            orig_h, orig_w = results.orig_shape

            result_items = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                class_id = int(box.cls[0])
                label = CLASSES[class_id]

                x_pct = x1 / orig_w * 100
                y_pct = y1 / orig_h * 100
                w_pct = (x2 - x1) / orig_w * 100
                h_pct = (y2 - y1) / orig_h * 100

                result_items.append({
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "score": score,
                    "value": {
                        "x": x_pct,
                        "y": y_pct,
                        "width": w_pct,
                        "height": h_pct,
                        "rectanglelabels": [label],
                    },
                })

            predictions.append({"result": result_items, "score": 0.0})

        return predictions
