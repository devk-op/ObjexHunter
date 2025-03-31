
import os
import shutil
from glob import glob
from pathlib import Path
from PIL import Image

def convert_to_yolo_format(user_dir='user_uploads', output_dir='custom_dataset'):
    class_names = [d for d in os.listdir(user_dir) if os.path.isdir(os.path.join(user_dir, d))]
    class_map = {cls: idx for idx, cls in enumerate(class_names)}

    # Prepare directories
    images_dir = os.path.join(output_dir, 'images', 'train')
    labels_dir = os.path.join(output_dir, 'labels', 'train')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for cls in class_names:
        img_paths = glob(os.path.join(user_dir, cls, '*'))
        for img_path in img_paths:
            try:
                img = Image.open(img_path)
                w, h = img.size
                bbox = [0.5, 0.5, 1.0, 1.0]  # full image box
                label_line = f"{class_map[cls]} {' '.join(map(str, bbox))}"

                # Save image
                fname = Path(img_path).stem
                new_img_path = os.path.join(images_dir, f"{fname}.jpg")
                img.convert("RGB").save(new_img_path, "JPEG")

                # Save label
                with open(os.path.join(labels_dir, f"{fname}.txt"), 'w') as f:
                    f.write(label_line + "\n")
            except Exception as e:
                print(f"[WARNING] Skipped {img_path}: {e}")

    # Create data.yaml
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write("path: " + str(Path(output_dir).resolve()) + "\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n")
        f.write("names: [" + ", ".join([f"'{c}'" for c in class_names]) + "]\n")

    print(f"[INFO] YOLO-format dataset created at: {output_dir}")
    print(f"[INFO] Classes: {class_map}")

if __name__ == "__main__":
    convert_to_yolo_format()
