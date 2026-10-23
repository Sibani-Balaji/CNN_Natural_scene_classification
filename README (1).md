# Natural Scene Classification with CNN and MobileNetV2

A deep learning project that classifies natural and urban scene images into six categories using TensorFlow/Keras. It compares a **custom CNN trained from scratch** with a **transfer-learning model built on MobileNetV2**, and evaluates both on a held-out test set.

## Results

| Model | Trainable Parameters | Test Accuracy |
|---|---|---|
| Custom CNN | 258,118 | 85.47% |
| MobileNetV2 (frozen backbone) | 7,686 | **90.00%** |

MobileNetV2 outperformed the custom CNN by about 4.5 percentage points while training far fewer parameters.

### Per-class performance (Custom CNN)

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Buildings | 0.8409 | 0.8467 | 0.8438 |
| Forest | 0.9848 | 0.9599 | 0.9722 |
| Glacier | 0.8899 | 0.7016 | 0.7846 |
| Mountain | 0.7415 | 0.8743 | 0.8024 |
| Sea | 0.8324 | 0.9059 | 0.8676 |
| Street | 0.8811 | 0.8583 | 0.8696 |

Forest is the easiest class to recognise. Glacier and mountain are most often confused with each other because of their similar snow, rock and sky textures.

## Dataset

[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) dataset with six classes: `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.

| Split | Images |
|---|---|
| Training | 11,228 (80% of `seg_train`) |
| Validation | 2,806 (20% of `seg_train`) |
| Test | 3,000 (`seg_test`) |
| Prediction (unlabelled) | 7,301 (`seg_pred`) |

Images are resized to 150 x 150 and loaded in batches of 32.

## Model Architectures

**Custom CNN**
- 4 convolution blocks (3x3 Conv + ReLU + 2x2 MaxPooling) with 32, 64, 128 and 128 filters
- Global Average Pooling
- Dense (128, ReLU), Dropout (0.4)
- Dense (6, Softmax)

**MobileNetV2 transfer learning**
- ImageNet-pretrained MobileNetV2 as a frozen feature extractor
- Data augmentation: random horizontal flip, rotation (0.08), zoom (0.10)
- Global Average Pooling, Dropout (0.3), Dense (6, Softmax)

## Training Setup

- Optimizer: Adam, loss: sparse categorical cross-entropy
- Callbacks: `ModelCheckpoint` (best validation accuracy), `EarlyStopping` (best weights restored), and `ReduceLROnPlateau` for MobileNetV2
- Custom CNN: up to 15 epochs. MobileNetV2: up to 10 epochs, learning rate 0.001
- Trained on a GPU in Google Colab

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Sibani-Balaji/CNN_Natural_scene_classification.git
   cd CNN_Natural_scene_classification
   ```
2. Download the dataset from Kaggle and upload the zip to your Google Drive (the notebook expects `Natural_Scene_Classification/archive (1).zip`), or change the path in the notebook.
3. Open the notebook in Google Colab, select a GPU runtime, and run all cells.

### Requirements

- Python 3
- TensorFlow / Keras
- NumPy, Matplotlib, scikit-learn

## Future Improvements

- Fine-tune the top layers of MobileNetV2 at a low learning rate
- Train at 224 x 224 to match the pretrained weights
- Generate per-class metrics and a confusion matrix for MobileNetV2
- Try stronger backbones such as EfficientNet or ResNet
- Run the best model on the unlabelled `seg_pred` images and build a simple web demo

## Author

**Sibani B**
Register No: RA2311026050056
Class: AIML - B
