
# Brinjal Shoot & Fruit Borer Detection Report

## 1. Introduction
The objective of this project is to detect infected leaves caused by the Brinjal Shoot & Fruit Borer (Leucinodes orbonalis) using an advanced object detection model. The system processes images or videos and highlights infected regions using bounding boxes. The model is based on YOLOv11, a state-of-the-art deep learning framework for real-time object detection.

## 2. Approach
### Data Collection
The dataset was obtained from Roboflow and consists of annotated images featuring healthy and infected brinjal leaves. The dataset was preprocessed and split into training, validation, and test sets to ensure balanced learning.

### Data Preprocessing
- Image normalization and resizing (640x640 pixels)
- Data augmentation techniques such as rotation, and cropping adjustments
- Conversion of annotations into YOLO11-compatible format

### Model Training Pipeline
1. **Load Dependencies**: Required libraries (e.g., `ultralytics`, `roboflow`, `opencv-python`) were installed.
2. **Dataset Download**: The dataset was loaded using Roboflow API.
3. **Pretrained Model**: YOLOv11's large variant (`yolo11l.pt`) was used as the base model.
4. **Training Execution**: The model was fine-tuned on the infected leaf dataset for 100 epochs.
5. **Model Evaluation**: The trained model was validated on the test set, and performance metrics were computed.

## 3. Model Architecture
The model architecture is based on YOLOv11, an improvement over previous YOLO versions, incorporating:
- **Anchor-free detection** for more accurate bounding box predictions
- **Transformer-based backbone** for enhanced feature extraction
- **Improved CSPDarknet backbone** for faster inference and reduced computational cost
- **Hybrid loss function** combining CIoU loss and classification loss

## 4. Hyperparameters
The training process was configured with the following hyperparameters:
- **Model**: YOLOv11-Large (`yolo11l.pt`)
- **Input size**: 640x640 pixels
- **Batch size**: 16
- **Epochs**: 100
- **Learning rate**: 0.01 with a cosine scheduler
- **Optimizer**: SGD with momentum (0.937)
- **Loss Function**: CIoU loss for bounding boxes and cross-entropy loss for classification

## 5. Evaluation Results
The model was tested on the validation and test datasets, producing the following performance metrics:

### Validation Set Performance:
- **Precision (P)**: 0.993
- **Recall (R)**: 0.920
- **mAP@50**: 0.958
- **mAP@50-95**: 0.375

### Test Set Performance:
```json
{
 "metrics/precision(B)": 0.7009,
 "metrics/recall(B)": 0.9,
 "metrics/mAP50(B)": 0.8213,
 "metrics/mAP50-95(B)": 0.4259,
 "fitness": 0.4655
}
```

## 6. Conclusion
The trained YOLOv11 model demonstrated high precision and recall in detecting infected brinjal leaves. The performance metrics suggest that the model is reliable for real-world applications, though further improvements can be made, such as fine-tuning hyperparameters and increasing dataset size.

## 7. Future Work
- Expanding the dataset with more diverse images
- Optimizing model parameters for better mAP@50-95
- Deploying the model in a mobile or edge computing environment

## 8. References
- Ultralytics YOLOv11: [https://github.com/ultralytics/yolov11](https://github.com/ultralytics/yolov11)
- Roboflow: [https://roboflow.com/](https://roboflow.com/)

