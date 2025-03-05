# Brinjal Shoot & Fruit Borer Detection

## Overview
This project aims to detect infected leaves caused by the Brinjal Shoot & Fruit Borer using an object detection model. The model is trained to identify abnormal sections in input images or videos, highlighting infected areas using bounding boxes.

## Folder Structure
```
Brinjal_ShootFruitBorer_detection/
│── input_image/        # Contains input images
│── output_image/       # Contains predicted images with bounding boxes
│── Complete_notebook.ipynb  # Jupyter notebook for training and evaluation
│── best.pt             # Trained model weights
│── predict_image.py    # Script to predict on images
│── predict_video.py    # Script to predict on videos
│── requirements.txt    # List of dependencies
```

## Model Training
The model is trained using YOLOv11. Below are the steps taken:

### 1. Install Dependencies
```bash
pip install ultralytics roboflow opencv-python
```

### 2. Load Dataset from Roboflow
```python
from roboflow import Roboflow
rf = Roboflow(api_key="hZbMXOP2oUimKaS4NBhf")
project = rf.workspace("self-jouzf").project("infected-leaf-detection")
version = project.version(1)
dataset = version.download("yolov11")
```

### 3. Download YOLOv11 Pretrained Model
```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt
```

### 4. Train the Model
```bash
yolo task=detect mode=train model=/content/yolo11l.pt \
     data=/content/Infected-leaf-detection-1/data.yaml \
     epochs=100 imgsz=640
```

### 5. Evaluate Model Performance
```python
from ultralytics import YOLO
model = YOLO('/content/runs/detect/train/weights/best.pt')
metrics = model.val(data='/content/Infected-leaf-detection-1/data.yaml', split="test")
print(metrics.results_dict)
```

## Model Evaluation Results
The model was trained for 100 epochs, achieving the following results:
- **Precision (P)**: 0.993
- **Recall (R)**: 0.920
- **mAP@50**: 0.958
- **mAP@50-95**: 0.375

Test results:
```json
{
 "metrics/precision(B)": 0.7009,
 "metrics/recall(B)": 0.9,
 "metrics/mAP50(B)": 0.8213,
 "metrics/mAP50-95(B)": 0.4259,
 "fitness": 0.4655
}
```

## Prediction on New Data

### Image Prediction
To detect infected areas in images and save the results in `output_image/`:
```bash
python predict_image.py
```

![output](https://github.com/user-attachments/assets/707a0743-6872-47ed-9c91-d580f4073d73)
<br/>


![output1](https://github.com/user-attachments/assets/2142352b-c836-4c3a-9c46-2a6125d05fcc)
<br/>

![output2](https://github.com/user-attachments/assets/d4d31d85-7c8f-4c5f-a24d-378a7bb281a4)
<br/>



### Video Prediction
To detect infected areas in videos and save the output in `output_video/`:
```bash
python predict_video.py
```

## Dependencies
All dependencies are listed in `requirements.txt`:
```
ultralytics
roboflow
opencv-python
```
Install them using:
```bash
pip install -r requirements.txt
```

## Authors
- **Asif Pervez Polok** 



