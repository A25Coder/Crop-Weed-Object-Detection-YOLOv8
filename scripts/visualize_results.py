import os
import cv2
import matplotlib.pyplot as plt

# Force look into the exact directory YOLO printed out
predict_folder = "/content/runs/detect/predict"
saved_files = os.listdir(predict_folder)

print(f"Files found in predict folder: {saved_files}")

# Grab the first file in that folder and display it
if len(saved_files) > 0:
    target_image_path = os.path.join(predict_folder, saved_files[0])

    img = cv2.imread(target_image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title("🎉 Custom YOLOv8 Crop & Weed Detection Result 🎉")
    plt.show()
else:
    print("Folder is empty. Try checking '/content/runs/detect/predict2'")
