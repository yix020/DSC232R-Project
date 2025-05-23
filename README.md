
# Polish Road Object Detection – Model Training and Evaluation

This project applies deep learning techniques to the [Traffic Road Object Detection Polish 12k dataset](https://www.kaggle.com/datasets/mikoajkoek/traffic-road-object-detection-polish-12k). It includes preprocessing, model development, and evaluation across training, validation, and test sets.

## Data Preprocessing

Key preprocessing steps included:

- **Image Resizing** to 128×128 resolution
- **Pixel Normalization** to [-1, 1] using mean=0.5, std=0.5 for each RGB channel
- **Class Distribution Analysis** with bar plots to identify imbalance
- **Class Weight Computation** for weighted loss in training
- **Label Validation** script to check format issues, orphaned/missing files, and class range
- **Train/Val/Test Split** following the provided YOLO directory structure

All image-label pairs were confirmed to be valid and in the correct format. No files were removed.

## Models Trained

Two classification models were implemented:

### 1. Simple CNN

- Custom 3-layer CNN built from scratch using PyTorch
- Underperformed with ~18% validation accuracy
- Showed signs of **underfitting**

### 2. Pretrained ResNet18

- Fine-tuned using ImageNet weights
- Achieved **training accuracy of 99.15%** and **test accuracy of 50.60%**
- Outperformed the custom CNN, but showed signs of **overfitting**

All experiments are documented in:  
📓 [`Milestone 3.ipynb`](./Milestone%203.ipynb)

## Evaluation Results

| Metric           | Value    |
|------------------|----------|
| Train Accuracy   | 99.15%   |
| Validation Accuracy | ~52%  |
| Test Accuracy    | 50.60%   |
| Test Error       | 49.40%   |

## Where the Model Fits on the Fitting Graph

The pretrained ResNet18 shows strong performance on training data but weaker generalization on test data. This places it in the **overfitting** zone of the model complexity vs. error curve.

## Conclusion

The ResNet18 model demonstrated the ability to learn complex patterns with high training accuracy. However, the significant gap between training and test performance indicates overfitting. To improve results:
- Add **data augmentation** (flips, crops, brightness)
- Apply **dropout** or **regularization**
- Consider deeper models (e.g., ResNet34) or more efficient ones (e.g., EfficientNet-B0)
- Explore **early stopping** and **learning rate scheduling**

## Repository Structure

- `Milestone 3.ipynb`: Final training, evaluation, and model comparison (CNN and ResNet)
- `setupAndEDA.ipynb`: Initial dataset exploration, class distribution, and validation
- `Road_Detection_Data/`: Clean YOLO-format image/label dataset

