# DeepVision AI — Project Report

## Abstract
DeepVision AI is an educational visual-inspection application combining supervised image classification with reference-based anomaly localization. It uses a transfer-learning ResNet18 classifier for known classes and a pretrained CNN feature-distance baseline to highlight spatial differences from normal examples.

## Problem statement
Manual visual inspection can be repetitive and inconsistent. A computer-vision assistant can help prioritize images for human review by classifying known categories and highlighting unusual regions.

## Objectives
- Build a browser-based interface for image upload and analysis.
- Use a deep neural network for supervised classification.
- Use pretrained deep features for anomaly scoring and heatmap visualization.
- Compare the two approaches and communicate limitations.

## System modules
1. Streamlit user interface
2. Image preprocessing
3. ResNet18 classifier training and inference
4. Normal-reference feature extraction
5. Anomaly score and heatmap generation
6. Result presentation and troubleshooting

## Methodology
### Classification
A pretrained ResNet18 is used as a feature extractor. The final fully connected layer is replaced to match the dataset classes. The final layer is trained using labeled training images, then evaluated on separate validation images.

### Anomaly localization
Normal reference images are passed through a pretrained ResNet18 feature extractor. A spatial mean and standard deviation feature template is computed. For a test image, normalized feature deviations from the template are aggregated into a spatial score map and resized to image dimensions for visualization.

## Evaluation plan
For classification, report validation accuracy and ideally precision, recall, F1-score, and a confusion matrix on a separate held-out test set. Do not report results until the model has actually been trained and tested.
For anomaly detection, qualitatively inspect heatmaps and, if pixel-level masks are available, compute an appropriate localization metric. Without ground-truth masks, describe the heatmap as a visual aid rather than a measured defect boundary.

## Limitations
- The supervised classifier requires representative labeled data.
- The anomaly baseline is sensitive to lighting, camera angle, background, and product alignment.
- The anomaly score is relative and not a calibrated probability.
- A pretrained model may not represent the chosen industrial domain well.
- The prototype is not certified for safety-critical quality control.

## Future work
- Fine-tune the backbone with more domain data.
- Add threshold calibration and a review queue.
- Add Grad-CAM for classifier explanations.
- Evaluate against a held-out test set.
- Compare with dedicated anomaly-detection methods such as PatchCore.
