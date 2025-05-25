
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
- Underperformed with ~2.45% validation accuracy
- Showed signs of **underfitting**

### 2. Pretrained ResNet18

- Fine-tuned using ImageNet weights
- Achieved **training accuracy of 98.50%** and **test accuracy of 50.60%**
- Outperformed the custom CNN, but showed signs of **overfitting**

All experiments are documented in:  
📓 [`Milestone 3.ipynb`](./Milestone%203.ipynb)

## Evaluation Results

| Metric           | Value    |
|------------------|----------|
| Train Accuracy   | 98.50%   |
| Train Error      | 1.50%    |
| Validation Accuracy | ~52%  |
| Test Accuracy    | 50.60%   |
| Test Error       | 49.40%   |


## Where the Model Fits on the Fitting Graph

Our model fits in the overfitting region of the fitting graph. The training error is extremely low (1.5%), while the test error is relatively high (49.40%), indicating the model performs very well on the training data but struggles to generalize to unseen data. This gap between train and test performance suggests the model has overfit to the training set.

## What are the next models you are thinking of and why?

Given that the current model (ResNet18) shows signs of overfitting, the next models I’m considering include deeper or more regularized architectures, such as ResNet34 or EfficientNet-B0. These models may capture richer features while incorporating improved regularization. Additionally, I’m considering applying dropout layers, data augmentation, or using early stopping to reduce overfitting. Exploring transfer learning with fine-tuning more layers could also help adapt pretrained models more effectively to this dataset.

## Conclusion

Two models were tested: a custom-built Simple CNN and a pretrained ResNet18. The Simple CNN showed signs of underfitting, with low training performance and validation accuracy stuck around 2.45%, indicating that the model lacked the capacity to learn meaningful features. The pretrained ResNet18 performed significantly better, achieving a low training error (1.5%) and a higher validation accuracy (~51%), but still showed signs of overfitting, with a high test error (49.40%). To improve generalization, future steps could include applying data augmentation, dropout, or early stopping, as well as exploring deeper architectures like ResNet34 or more efficient models such as EfficientNet-B0. These adjustments may help balance model capacity and regularization to reduce overfitting while maintaining strong performance.

## Repository Structure

- `Milestone 3.ipynb`: Final training, evaluation, and model comparison (CNN and ResNet)
- `setupAndEDA.ipynb`: Initial dataset exploration, class distribution, and validation
- `Road_Detection_Data/`: Clean YOLO-format image/label dataset

