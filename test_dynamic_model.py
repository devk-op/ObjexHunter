from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test Trained YOLOv8 Model")
    parser.add_argument('--model', type=str, default='model_dynamic.pt', help='Path to trained YOLOv8 model')
    parser.add_argument('--source', type=str, default='0', help='0 for webcam, or path to image/video')
    args = parser.parse_args()

    # Load model
    model = YOLO(args.model)

    # Predict
    model.predict(source=args.source, show=True)

if __name__ == "__main__":
    main()
