import os
import shutil
from sklearn.model_selection import train_test_split

# 1. Automatically find where the images actually are
BASE_DIR = "local_dataset"
DATA_DIR = None

for root, dirs, files in os.walk(BASE_DIR):
    image_files = [f for f in files if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
    if len(image_files) > 0:
        DATA_DIR = root
        IMAGES = image_files
        break

if DATA_DIR is None:
    raise FileNotFoundError("Could not find any .jpeg, .jpg, or .png images inside local_dataset.")

print(f"🎯 Success! Found {len(IMAGES)} images inside folder: {DATA_DIR}")
print("Sorting images into YOLO format splits...")

# 2. Split into 80% Train, 20% Validation
train_imgs, val_imgs = train_test_split(IMAGES, test_size=0.2, random_state=42)

def setup_yolo_dir(images, subset):
    os.makedirs(f"yolo_data/{subset}/images", exist_ok=True)
    os.makedirs(f"yolo_data/{subset}/labels", exist_ok=True)
    for img in images:
        shutil.copy(os.path.join(DATA_DIR, img), f"yolo_data/{subset}/images/{img}")
        base_name = os.path.splitext(img)[0]
        lbl = base_name + ".txt"
        if os.path.exists(os.path.join(DATA_DIR, lbl)):
            shutil.copy(os.path.join(DATA_DIR, lbl), f"yolo_data/{subset}/labels/{lbl}")

setup_yolo_dir(train_imgs, "train")
setup_yolo_dir(val_imgs, "val")

# 3. Create the dataset configuration file (Fixed Attribute Error by forcing import)
import os as safe_os
absolute_path = safe_os.path.abspath('yolo_data')

yaml_content = f"""
path: {absolute_path}
train: train/images
val: val/images

names:
  0: crop
  1: weed
"""
with open("dataset.yaml", "w") as f:
    f.write(yaml_content)

# 4. Initialize and train the YOLO model
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

# Run the training loop
model.train(
    data='dataset.yaml',
    epochs=15,
    imgsz=512,          # Enforces the mandatory 512x512 resolution constraint [cite: 30, 39]
    batch=16,
    augment=True,       # Handles dataset expansion via built-in transformations [cite: 41]
    device=0            # Processes variables natively on your free cloud GPU
)

print("\n🎉 Success! Your Project 5 model is training.")