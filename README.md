# Polish Road Object Detection – Data Exploration & Preprocessing

This project focuses on preparing and analyzing the [Traffic Road Object Detection Polish 12k dataset](https://www.kaggle.com/datasets/mikoajkoek/traffic-road-object-detection-polish-12k) for training an object detection model. The dataset includes thousands of annotated traffic images captured under various road conditions, with YOLO-format bounding box labels.

## Data Exploration

The dataset was explored in terms of:

- Total image and annotation counts across `train`, `valid`, and `test` splits
- Object class distribution using bar plots
- Sample bounding box visualizations with class labels
- Image width and height distribution to assess input variability
- Missing image/label detection and cleanup of orphaned files

See the full notebook here: [`setupAndEDA.ipynb`](./setupAndEDA.ipynb)

## Preprocessing Pipeline

To prepare the dataset for training, we performed the following preprocessing steps:

- **Image Resizing with Padding**: All images were resized to `640×640` using a custom `resize_with_padding()` function, which preserves the original aspect ratio and pads with black borders. This ensures consistent input dimensions without distorting object shapes — a crucial factor for object detection.
- **Pixel Normalization**: Each image was normalized to the `[0, 1]` range by dividing all pixel values by 255.
- **Missing Label/Image Cleanup**: We identified and removed over 7,000 orphaned label files that had no matching image, ensuring training stability and data integrity.

## Repository Contents

- `setupAndEDA.ipynb`: Jupyter Notebook containing the full exploration and preprocessing workflow
- `resize_with_padding()`: Python function used for aspect-ratio preserving image resizing
- All original images and labels remain in YOLO format under the respective split folders

## Next Steps

With data cleaned and standardized, the next phase involves defining a YOLO-compatible dataset config and launching training with a model like YOLOv5 or YOLOv8.


