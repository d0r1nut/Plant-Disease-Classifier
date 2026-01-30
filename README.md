# Plant Disease Classifier with Explainable AI

A deep learning project for identifying plant diseases from photos. Built with ResNet34, achieving ~95% accuracy, and includes visualizations to show what the model is actually looking at when it makes predictions.

## Overview

This project classifies plant diseases using transfer learning and provides explainability through Grad-CAM visualizations. The model weights are pre-trained and included, so you can run inference immediately. The codebase also includes a complete training pipeline if you want to retrain on your own data.

Key features:
- Pre-trained ResNet34 model with ~95% accuracy
- Grad-CAM heatmaps to visualize which leaf regions the model uses for diagnosis
- SQLite database logging for all predictions
- Two-stage training approach (frozen backbone → fine-tuning)
- Handles class imbalance with weighted sampling

## What's In Here

**Transfer Learning with a Two-Stage Approach** — Started with a frozen ResNet34 backbone, trained just the new layers first, then unfroze everything and fine-tuned. Standard stuff but it works.

**Grad-CAM Visualizations** — The heatmaps show you exactly where on the leaf the model was looking when it said "yep, that's powdery mildew." Way better than a black box.

**Handles Class Imbalance** — Used weighted sampling and data augmentation. Some diseases had way more images than others, so I had to be careful about that.

**SQLite Logging** — Every inference gets written to the database with the prediction, confidence score, and filename. Makes it easy to track which images the model struggled with.

## How It Works

**The Model** — ResNet34 pretrained on ImageNet. Training uses the Adam optimizer with a learning rate scheduler that reduces learning rate when validation loss plateaus. Input images are normalized, augmented with rotation and color adjustments, and resized to 224×224.

**The Pipeline** — Images are normalized and passed through the model. Grad-CAM extracts gradients from deeper layers to generate attention heatmaps. All predictions are logged to SQLite with confidence scores and metadata for later analysis.

**Class Imbalance** — Weighted sampling and data augmentation account for uneven disease distribution in the dataset.

## Results

Achieved **95.59% accuracy** on the test set with training loss: 0.1562, validation loss: 0.1536.

The heatmaps effectively highlight disease-specific features. For example, Early Blight predictions focus on brown lesion patterns while ignoring background elements, indicating the model learns authentic plant pathology rather than superficial image artifacts.

## Getting Started

**Requirements:**
- Python 3.9+
- GPU recommended for training (CPU works for inference)

**Setup:**

1. Clone the repository or download the files

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Prepare the dataset:
    The dataset is excluded from version control. Download plant disease images from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) and place them in a `dataset/` directory.

4. View dataset statistics:
    ```bash
    python data_stats.py
    ```
    Shows class distribution and generates charts.

5. Train the model (optional—pre-trained weights are included):
    ```bash
    python train_advanced.py
    ```
    Implements two-stage training: frozen backbone first, then fine-tuning.

6. Run inference on a new image:
    ```bash
    python inference.py
    ```
    Generates predictions with confidence scores and saves a Grad-CAM heatmap overlay.

7. Evaluate on the full test set:
    ```bash
    python evaluate.py
    ```
    Produces a confusion matrix and detailed performance metrics.

8. View prediction history:
    ```bash
    python check_db.py
    ```
    Displays all logged predictions from prior inference runs.

## Project Structure

```
├── train_advanced.py          # Two-stage training script
├── inference.py               # Run predictions on new images
├── evaluate.py                # Generate confusion matrix and metrics
├── grad_cam.py                # Grad-CAM visualization implementation
├── data_stats.py              # Dataset statistics and distribution charts
├── check_db.py                # View inference prediction logs
├── plant_doctor_model_v2.pth  # Pre-trained model weights
├── plant_disease.db           # SQLite database (created at runtime)
├── requirements.txt           # Python dependencies
├── Dockerfile                 # CPU container image
├── Dockerfile.gpu             # GPU container image
└── README.md
```