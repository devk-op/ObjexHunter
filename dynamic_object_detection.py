
import os
import argparse
import shutil
from pathlib import Path
from glob import glob
from ultralytics import YOLO
import cv2

def convert_to_yolo_format(input_dir='user_uploads', output_dir='custom_dataset'):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    images_dir = output_path / 'images' / 'train'
    labels_dir = output_path / 'labels' / 'train'

    # Reset output directory
    if output_path.exists():
        shutil.rmtree(output_path)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Map class names to indices
    class_names = [d.name for d in input_path.iterdir() if d.is_dir()]
    class_map = {name: idx for idx, name in enumerate(class_names)}

    # Copy images and create YOLO labels
    for class_name, class_id in class_map.items():
        for img_path in glob(os.path.join(input_path, class_name, "*.[jp][pn]g")):
            img_path = Path(img_path)
            new_name = f"{class_name}_{img_path.name}"
            new_img_path = images_dir / new_name
            new_label_path = labels_dir / (new_img_path.stem + ".txt")

            # Copy image
            shutil.copy(img_path, new_img_path)

            # Create fake full-image bounding box for now
            with open(new_label_path, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

    # Write data.yaml
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_path.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n")
        f.write("names: [" + ", ".join(f"'{name}'" for name in class_names) + "]\n")

    print(f"[INFO] YOLO dataset created at: {output_path}")
    print(f"[INFO] Classes: {class_map}")
    print(f"[INFO] data.yaml generated at: {yaml_path}")
    return str(yaml_path)

def fine_tune_model(dataset_yaml_path, model_output_path):
    print(f"[INFO] Fine-tuning YOLOv8 model on: {dataset_yaml_path}")
    model = YOLO("yolov8n.pt")  # You can change this to yolov8s.pt etc.
    model.train(data=dataset_yaml_path, epochs=20, imgsz=640)
    model.save(model_output_path)
    print(f"[INFO] Model saved to {model_output_path}")

def main():
    parser = argparse.ArgumentParser(description="Dynamic Object Detection with YOLOv8")
    parser.add_argument('--input_dir', type=str, default='user_uploads', help='User-uploaded class folders')
    parser.add_argument('--output_model', type=str, default='model_dynamic.pt', help='Path to save trained model')
    args = parser.parse_args()

    # Step 1: Convert to YOLO dataset
    dataset_yaml_path = convert_to_yolo_format(args.input_dir, 'custom_dataset')

    # Step 2: Train YOLO model
    fine_tune_model(dataset_yaml_path, args.output_model)

if __name__ == "__main__":
    main()
