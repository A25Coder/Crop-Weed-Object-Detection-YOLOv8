<img width="790" height="812" alt="image" src="https://github.com/user-attachments/assets/86a5a00a-4b96-4e89-9ff2-57aea1170685" /># Automated Crop & Weed Object Detection Pipeline (YOLOv8)

An end-to-end computer vision pipeline utilizing the YOLOv8 architecture to automate precision agricultural crop tracking and weed identification. This project transitions loose field imagery into structured, GPU-ready tensors for high-speed spatial inference.

## 🚀 Key Performance Deliverables
* **Real-Time Inference Speed:** ~11.6ms per frame, enabling edge-device field deployments.
* **Peak Target Confidence:** Achieved up to 90% confidence mapping for crop classification.
* **Target Categories:** Class 0 (Crop / Sesame) and Class 1 (Weed / Impurities).

## 🛠️ Technical Stack & Frameworks
* **Core Language:** Python
* **Computer Vision & Deep Learning:** Ultralytics YOLOv8, PyTorch, OpenCV
* **Data Processing:** Scikit-Learn, Shutil, OS
* **Hardware Acceleration:** Cloud-based GPU infrastructure (Google Colab)

## 📁 Repository Structure

* `scripts/train_yolo.py`: Python module automating the 80/20 train/validation partitioning, directory construction, and training loops.
* `scripts/visualize_results.py`: An evaluation utility using OpenCV and Matplotlib to extract trained model inference outputs and visually display bounding boxes.

## ⚙️ Model Pipeline & Core Workflow
1. **Automated Scanning:** The script dynamically crawls data structures to identify `.jpeg`, `.jpg`, and `.png` image matrices.
2. **80/20 Partitioning:** Images and matching raw text coordinate files are randomly split into distinct Training (80%) and Validation (20%) datasets to ensure rigorous testing.
3. **Matrix Alignment:** Images are systematically scaled to a mandatory $512 \times 512$ pixel grid profile to preserve bounding box aspect ratios.
4. **Data Augmentation:** Embedded transformations (spatial transformations, contrast adjustments, and flipping) are executed at runtime to scale training variations and protect against overfitting.
5. **Transfer Learning:** Training initializes from a pretrained `yolov8n.pt` base model, repurposing underlying visual patterns for agricultural feature extraction over 15 epochs with a batch size of 16.



<img width="790" height="812" alt="image" src="https://github.com/user-attachments/assets/b666ba43-55af-4962-824c-d1d5aa4db150" />



