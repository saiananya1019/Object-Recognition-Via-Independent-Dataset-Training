Object Recognition via Independent Dataset Training

📌 Overview

This project focuses on object recognition using a custom-trained YOLOv8 model. Unlike traditional approaches that rely on pre-trained datasets, this system independently collects and labels data for training. The goal is to accurately identify objects such as jackets, gloves, goggles, helmets, and footwear.

🚀 Features

Custom dataset collection and annotation

YOLOv8 model training on an independent dataset

Real-time object detection

Google Colab integration for training

Evaluation metrics for performance analysis

🛠️ Tech Stack

Deep Learning Framework: YOLOv8 (Ultralytics)

Programming Language: Python

Libraries: OpenCV, TensorFlow/PyTorch

Dataset Labeling: LabelImg

Training Environment: Google Colab (T4 GPU)

📂 Dataset Details

Objects Tracked: Jackets, Gloves, Goggles, Helmets, Footwear

Annotation Tool: LabelImg

Data Format: YOLO format (txt annotations)

Storage: Google Drive (for Colab training)

⚙️ Installation & Usage

1️⃣ Clone the Repository
git clone https://github.com/your-username/Object-Recognition-Via-Independent-Dataset-Training.git
cd Object-Recognition-Via-Independent-Dataset-Training
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Train the Model
Run the YOLOv8 training script on Google Colab:
!yolo task=detect mode=train model=yolov8s.pt data=config.yaml epochs=50 imgsz=640
4️⃣ Run Object Detection
!yolo task=detect mode=predict model=runs/train/exp/weights/best.pt source=images/

📊 Results & Performance

Training Accuracy

Validation Accuracy

Inference Speed

Confusion Matrix & Loss Graphs

🤝 Contributing
Feel free to contribute by submitting pull requests.

Refer The research paper we have made for this project.
