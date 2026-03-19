from label_studio_ml.api import init_app
from yolo_backend import YOLOBackend

app = init_app(model_class=YOLOBackend)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9090)
