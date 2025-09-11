# Bone Anomaly Detection on MURA Dataset

## Project Description

This project focuses on automated bone anomaly detection from X-ray images using deep learning. The approach involves a two-stage process developed and demonstrated in Jupyter notebooks:

1. **Body Part Classification**: First, a model predicts which body part (wrist, elbow, etc.) the X-ray belongs to.
2. **Anomaly Detection**: Based on the predicted body part, a dedicated model assesses whether the image is normal or abnormal.
3. **Explainability**: Grad-CAM visualizations are provided for interpretable predictions, highlighting regions influencing the model's decisions.

Both stages use the ResNet-18 architecture. The implementation is modular, allowing extension to other architectures or datasets.

## Dataset

**MURA (Musculoskeletal Radiographs)** is one of the largest public radiographic image datasets, released by the Stanford ML Group. It consists of over 40,000 images from studies on seven upper extremity body parts — elbow, finger, forearm, hand, humerus, shoulder, and wrist. Each study is labeled as either normal or abnormal.  
*Note: The dataset is not provided in this repository. Please obtain it from the [Stanford ML Group MURA website](https://stanfordmlgroup.github.io/competitions/mura/).*

## How to Run

Execute the following notebooks in order:

1. **Train the models:**
   - Open `Separate Body Part and Classifier Training.ipynb`
   - Run all cells. This will:
     - Train the body part classifier model.
     - Train separate anomaly detection models for each body part.
     - Save the trained models for later use.

2. **Visualize predictions with Grad-CAM:**
   - Open `gradCAM.ipynb`
   - Run all cells. This notebook:
     - Loads a trained model.
     - Generates and displays Grad-CAM visualizations for selected X-ray images, illustrating which regions influenced each prediction.

## Requirements

- Python 3.8 or above
- PyTorch
- torchvision
- numpy
- matplotlib
- opencv-python
- pandas
- tqdm

Install dependencies with:
```bash
pip install torch torchvision numpy matplotlib opencv-python pandas tqdm
```
**Key Features:**
- Two-stage deep learning pipeline for enhanced accuracy
- ResNet-18 architecture implementation for both classification stages  
- Explainable AI through Grad-CAM visualizations
- Modular design allowing easy extension to other architectures
- Comprehensive evaluation on standard medical imaging dataset

**Workflow:**
1. **Body Part Classification**: First stage model identifies which body part (wrist, elbow, shoulder, finger, hand, humerus, forearm) is shown in the X-ray
2. **Specialized Anomaly Detection**: Based on the classified body part, a dedicated model trained specifically for that anatomy assesses whether the image shows normal or abnormal conditions
3. **Visual Explanation**: Grad-CAM generates heatmap visualizations highlighting the regions that most influenced the model's decision

## Architecture Details

### Stage 1: Body Part Classifier
- **Model**: ResNet-18 with modified final layer
- **Input**: Preprocessed X-ray image (224x224 pixels)
- **Output**: 7-class classification (body part prediction)
- **Training Strategy**: Multi-class classification with cross-entropy loss
- **Data Augmentation**: Random rotations, flips, and intensity adjustments

### Stage 2: Anomaly Detection Models  
- **Architecture**: Seven separate ResNet-18 models, one for each body part
- **Rationale**: Body part-specific models capture unique anatomical features and pathology patterns
- **Input**: X-ray image of identified body part
- **Output**: Binary classification (normal/abnormal)
- **Training Strategy**: Binary classification with class weighting for imbalance

### Grad-CAM Implementation
- **Target Layer**: Final convolutional layer (layer4) of ResNet-18
- **Visualization**: Class activation mapping overlaid on original images
- **Purpose**: Provides interpretability by highlighting regions contributing to predictions
- **Clinical Value**: Helps understand AI decision-making process for medical professionals

## Notes

- Please download the MURA dataset and place it in the specified directory as expected by the notebooks.
- Modify paths as needed in the notebook cells for your local setup.
- Results and figures will be saved or displayed as configured in the notebooks.

---
