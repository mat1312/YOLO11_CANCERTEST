# Skin Cancer Detection with YOLO11

This repository contains a step-by-step guide to set up and train a YOLO11 model for detecting various types of skin cancer using the Ultralytics library.

---

## Overview
This project utilizes YOLO11, an advanced object detection model, to perform fine-tuned training on a dataset of skin cancer images. The dataset is obtained from Roboflow and formatted for YOLOv8, with custom modifications to ensure compatibility with YOLO11.

---

## Getting Started

### Prerequisites
- Ensure you have access to a GPU for optimal performance.
  - Use the command `!nvidia-smi` to verify GPU availability.
  - If GPU is not enabled, navigate to **Edit -> Notebook settings -> Hardware accelerator**, select GPU, and save.

### Environment Setup
Install the required packages:
```bash
!pip install ultralytics roboflow
```

Set the working directory:
```python
import os
HOME = os.getcwd()
print(HOME)
```

---

## Installing YOLO11
Install YOLO11 via the Ultralytics library:
```bash
%pip install ultralytics
```
Check the installation:
```python
import ultralytics
ultralytics.checks()
```

---

## Dataset Preparation
1. **Download Dataset**:
   - Use the Roboflow API to download the dataset in YOLOv8 format:
   ```python
   from roboflow import Roboflow
   rf = Roboflow(api_key="<your_api_key>")
   project = rf.workspace().project("skin-cancer-recogniser")
   version = project.version(1)
   dataset = version.download("yolov11")
   ```
2. **Organize Dataset**:
   - Ensure the dataset is located in `{HOME}/datasets`.
   - Modify the `data.yaml` file for YOLO11 compatibility:
   ```bash
   !sed -i '$d' {dataset.location}/data.yaml   # Remove last lines
   !echo 'test: ../test/images' >> {dataset.location}/data.yaml
   !echo 'train: ../train/images' >> {dataset.location}/data.yaml
   !echo 'val: ../valid/images' >> {dataset.location}/data.yaml
   ```

---

## Training the Model
1. **Start Training**:
   ```bash
   !yolo task=detect mode=train model=yolo11s.pt data={dataset.location}/data.yaml epochs=10 imgsz=640 plots=True
   ```
2. **Training Outputs**:
   - Results, including logs and confusion matrices, are saved in `{HOME}/runs/detect/train/`.

---

## Validation
Validate the trained model:
```bash
!yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml
```
View results in `{HOME}/runs/detect/val/`.

---

## Inference
Perform predictions on test images:
```bash
!yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True
```
Results are saved in `{HOME}/runs/detect/predict/`.

---

## Visualizing Results
Use the following Python script to display results:
```python
import glob
from IPython.display import Image as IPyImage, display

latest_folder = max(glob.glob('/content/runs/detect/predict*/'), key=os.path.getmtime)
for img in glob.glob(f'{latest_folder}/*.jpg')[:3]:
    display(IPyImage(filename=img, width=600))
```

---

## Notes
- **Hardware Requirements**:
  - A GPU with sufficient memory (e.g., Tesla T4) is recommended.
- **Dataset Format**:
  - Ensure your dataset is in YOLO format and structured as follows:
    ```
    datasets/
      train/
        images/
        labels/
      valid/
        images/
        labels/
      test/
        images/
    ```

---

## Resources
- [Ultralytics Documentation](https://docs.ultralytics.com)
- [Roboflow Documentation](https://docs.roboflow.com)

---

## License
This project is licensed under the MIT License.
