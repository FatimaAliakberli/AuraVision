# AuraVision

> **Crowd Age Group Majority Classifier** — Milestone 1

AuraVision is a computer vision project designed to identify the **dominant age group within a crowd scene**. Rather than relying on facial recognition, the model learns to distinguish age groups through visual cues such as clothing style, accessories, body posture, and contextual belongings — making it more privacy-conscious and robust in real-world scenarios. The ultimate goal is to deploy AuraVision in settings such as public spaces, retail environments, or event venues where understanding the age composition of a crowd can inform decision-making.

---

## Dataset

Images were collected **manually from [Pexels](https://www.pexels.com/)**, a free stock photography platform, using targeted search queries to capture visually distinct representations of each age group:

| Class | Search Queries Used |
|---|---|
| **Children** | `children`, `children at school` |
| **Adults** | `business women`, `business men` |
| **Seniors** | `seniors`, `seniors walking` |

The dataset consists of **117 images** in total across three classes, with **19 images** reserved as test data containing mixed age groups for inference.

---

## Project Structure

```
AuraVision/
│
├── dataset/
│   ├── train/
│   │   ├── children/        # Training images — children
│   │   ├── adults/          # Training images — adults
│   │   └── seniors/         # Training images — seniors
│   └── test/                # Mixed test images (no subfolders)
│
├── result_images/                 # Output visualizations (auto-created)
│
├── train_model.py           # Training script → produces age_classifier.pth
├── visualize_results.py     # Inference + visualization script
├── age_classifier.pth       # Saved model weights (generated after training)
└── README.md
```

---

## Model Architecture

AuraVision uses **transfer learning** on top of **EfficientNet-B0** pretrained on ImageNet. Given the small dataset size, training from scratch would lead to severe overfitting — instead, the early layers (which already encode general visual features like textures, edges, and shapes) are frozen, and only the final feature blocks and a custom classification head are fine-tuned.

```
EfficientNet-B0 (pretrained, partial freeze)
        ↓
  Custom Classifier Head
  ├── Dropout (0.4)
  ├── Linear → 128
  ├── ReLU
  ├── Dropout (0.3)
  └── Linear → 3 classes (adults / children / seniors)
```

**Training strategy:**
- Heavy data augmentation (random crops, flips, rotation, color jitter) to compensate for the small dataset
- Weighted random sampling to handle class imbalance
- AdamW optimizer with cosine annealing scheduler
- 40 epochs, batch size 8

---

## Usage

### 1. Install Dependencies

```bash
pip install torch torchvision pillow numpy
```

### 2. Train the Model

```bash
python train_model.py
```

Reads from `dataset/train/`, trains for 40 epochs, and saves the best model to `age_classifier.pth`. Training and validation accuracy are printed each epoch.

### 3. Run Inference & Visualize

```bash
python visualize_results.py
```

Processes every image in `dataset/test/`, divides each into a **4×4 patch grid**, classifies each patch independently using the trained model, and saves color-coded results to `results/`.

**Color coding:**

| Color | Age Group |
|---|---|
| 🔵 Blue | Adults |
| 🟢 Green | Children |
| 🔴 Red | Seniors |

Each saved image includes a **top banner** showing the majority class and patch vote counts, plus a **legend** in the corner. Example output filename: `pexels-example-1234_result.jpg`

---

## Limitations — Milestone 1

The following limitations have been identified and will be addressed before the final submission:

**1. Small training dataset**
The model was trained on only 117 images, which is far below what is typically needed for robust generalization. For the final version, we plan to significantly expand the dataset across all three classes.

**2. Narrow search diversity**
Training data was sourced using specific queries (e.g. *business men*, *children at school*), meaning the model may struggle with age groups presented in different contexts — for example, a child riding a bicycle or an adult in casual clothing. We will broaden our data collection strategy to cover a wider range of real-world scenarios aligned with our deployment purpose.

**3. No quantitative evaluation metric**
Due to the absence of ground truth labels for test images, model performance was assessed purely through visual inspection of the patch-colored outputs. For the final submission, we plan to either manually label the test set or explore alternative evaluation strategies to compute a concrete accuracy or error rate.

---

## Future Work

- Expand dataset to 1000+ images per class with diverse contexts
- Introduce labeled test data for quantitative evaluation (accuracy, F1-score)
- Experiment with larger backbones (EfficientNet-B3, ResNet50) as data grows
- Explore sliding-window inference for higher spatial resolution in crowd scenes

---

## Acknowledgements

All images used in this project were sourced from [Pexels](https://www.pexels.com/) under their free-to-use license.