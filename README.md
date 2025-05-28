# Polish Road Object Detection – Data Exploration & Preprocessing

This project focuses on preparing and analyzing the [Traffic Road Object Detection Polish 12k dataset](https://www.kaggle.com/datasets/mikoajkoek/traffic-road-object-detection-polish-12k) for training an object detection model. The dataset includes thousands of annotated traffic images captured under various road conditions, with YOLO-format bounding box labels.

## Data Exploration

The dataset was explored in terms of:

- Total image and annotation counts across `train`, `valid`, and `test` splits
- Object class distribution using bar plots and percentages
- Sample bounding box visualizations with class labels
- Image width and height distribution to assess input variability
- Missing image/label detection and cleanup of orphaned files

See the full notebook here: [`setupAndEDA.ipynb`](./setupAndEDA.ipynb)

## Preprocessing Pipeline

To prepare the dataset for training, we performed the following preprocessing steps:

- **Image Resizing**: All images were resized to `252×252` resolution for uniformity across training.
- **Pixel Normalization**: Each image is grayscale and normalized with mean=0.5, std=0.5.
- **Missing Label/Image Cleanup**: We identified and removed 6,940 orphaned label files with no matching image to ensure dataset consistency.
- **Class Distribution**: A detailed bar chart was created to visualize class imbalance. Classes like `car` and `pedestrian` dominate the dataset.
- **Class Weights Calculation**: Based on class frequencies, inverse-proportional weights were computed for use in the loss function to counteract imbalance.
- **Data Augmentation**: Applied on-the-fly using PyTorch `transforms`, including:
  - Horizontal flips
  - Random brightness/contrast adjustments
  - Small rotations
- Augmented images are generated during training and not saved to disk.

## Repository Contents

- `setupAndEDA.ipynb`: Jupyter Notebook with full data exploration and preprocessing
- `README.md`: This file, documenting the preprocessing and next steps
- `images/` and `labels/` folders under `train/`, `valid/`, and `test/`

## Environment Setup & SLURM Configuration
This project was developed using a custom Python environment created locally for data exploration and preprocessing.

Environment Details:
Python Version: 3.10

Environment Manager: conda 

Main Libraries Used:

pandas

numpy

matplotlib

opencv-python (for cv2)

tqdm

os, glob, shutil (Python standard library)

To recreate this environment, run:

conda create -n tf_env python=3.10
conda activate tf_env
pip install pandas numpy matplotlib opencv-python tqdm
pip install torch==2.2.2 torchvision==0.17.2

## SLURM Job Configuration

SLURM was not used for preprocessing in this notebook. All data preparation was run locally.

## Next Steps

- Train a custom object detection model using the cleaned and augmented data
- Optionally integrate a custom PyTorch `Dataset` class to support YOLO label format
- Evaluate per-class performance and refine augmentation strategy if needed
