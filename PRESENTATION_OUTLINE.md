# Presentation Outline (8 slides)

1. **Title:** DeepVision AI — Intelligent Product Defect Inspection
2. **Problem:** manual inspection is repetitive; explain the target product category.
3. **Solution:** one application, two modes: classification and anomaly heatmap.
4. **Architecture:** uploaded image → preprocessing → CNN → prediction / anomaly map → UI.
5. **Mode A:** dataset folder structure, transfer learning, training and validation.
6. **Mode B:** normal reference images, feature comparison, relative anomaly score, heatmap.
7. **Demo and evaluation:** show actual test images, predictions, metrics, and failure cases. Only use metrics you actually measured.
8. **Limitations and future work:** data quality, lighting, domain shift, calibration, human review.

## 60-second explanation
“DeepVision AI is a visual inspection prototype. It combines two complementary approaches. Mode A is supervised classification: a ResNet18 model is adapted to labeled normal and defective examples. Mode B is reference-based anomaly detection: a pretrained CNN extracts spatial features from normal examples and highlights differences in a test image. The interface displays the prediction and heatmap. The system is an educational assistant, not a certified inspection tool, because results depend on the data and imaging conditions.”
