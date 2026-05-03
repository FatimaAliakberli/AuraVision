# AuraVision

> **Crowd Age Group Majority Classifier** — Milestone 1

AuraVision is a computer vision project designed to identify the **dominant age group within a crowd scene**. Rather than relying on facial recognition, the model learns to distinguish age groups through visual cues such as clothing style, accessories, body posture, and contextual belongings — making it more privacy-conscious and robust in real-world scenarios. The ultimate goal is to deploy AuraVision in settings such as public spaces, retail environments, or event venues where understanding the age composition of a crowd can inform decision-making.

## Table of Contributions 

| Team Member | Contribution |
| :---  | :--- |
| Fatima Alakbarli | Data Training |
| Shahd Elaydy | Data Collection |

## Dataset

Images were collected **manually from [Pexels](https://www.pexels.com/)**, a free stock photography platform, using targeted search queries to capture visually distinct representations of each age group:

| Class | Search Queries Used |
|---|---|
| **Children** | `children`, `children at school` |
| **Adults** | `business women`, `business men` |
| **Seniors** | `seniors`, `seniors walking` |

The dataset consists of **117 images** in total across three classes, with **19 images** reserved as test data containing mixed age groups for inference.


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
├── result_images/           # Output visualizations (auto-created)
│
├── train_model.py           # Training script → produces age_classifier.pth
├── visualize_results.py     # Inference + visualization script
├── age_classifier.pth       # Saved model weights (generated after training)
└── README.md
```


## Model Architecture

AuraVision uses **transfer learning** on top of **EfficientNet-B0** pretrained on ImageNet. Given the small dataset size, training from scratch would lead to severe overfitting; instead, the early layers (which already encode general visual features like textures, edges, and shapes) are frozen, and only the final feature blocks and a custom classification head are fine-tuned.

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

Each saved image includes a **top banner** showing the majority class and patch vote counts, plus a **legend** in the corner. Example output filename: `pexels-example-1234_result.jpg`


## Evaluation 
Evaluating the performance of AuraVision was challenging because the test images were **not labeled with ground truth age groups**. Without labeled test data, it was not possible to compute standard quantitative metrics such as **accuracy, precision, recall, or F1-score** on the test set.

To overcome this limitation, we implemented a **visualization-based evaluation approach** using the `visualize_results.py` script. This script divides each test image into a **4×4 grid of patches** and classifies each patch independently using the trained model. The predictions are then visualized directly on the image using colored overlays.

### Patch Prediction Visualization

Each patch is colored according to the predicted age group:

| Color | Predicted Age Group |
|---|---|
| 🟢 Green | Children |
| 🔵 Blue | Adults |
| 🔴 Red | Seniors |

After all patches are classified, the script also determines the **majority class** across the image and displays it in a banner at the top of the output image. Additionally, the number of patch votes for each class is shown to provide a clearer understanding of how the model reached its decision.

This visualization allowed us to **manually inspect the behavior of the model** and evaluate whether predictions appeared reasonable based on the visible content of the image.

### Observations

During training, the model achieved **over 90% accuracy on the training data**. While this indicates that the model successfully learned patterns from the dataset, such a high training accuracy combined with a relatively small dataset suggests a risk of **overfitting**.

By manually analyzing the visualization outputs on the test images, we observed the following:

- **Children were identified most reliably**, with approximately **70% of relevant patches classified correctly**. The model appeared to detect visual features commonly associated with children, such as smaller body proportions, school-related environments, and certain clothing patterns.
  
- **Adults and seniors were more difficult for the model to distinguish**. In several cases, the model predicted adults when the person appeared to be a senior, and vice versa.

- Many misclassifications occurred when individuals in the image had **similar clothing styles**, such as formal outfits, coats, or neutral-colored attire, which reduced the visual cues available for distinguishing between these age groups.


## Limitations — Milestone 1

The following limitations have been identified and will be addressed before the final submission:

**1. Small training dataset**
The model was trained on only 117 images, which is far below what is typically needed for robust generalization. For the final version, we plan to significantly expand the dataset across all three classes.

**2. Narrow search diversity**
Training data was sourced using specific queries (e.g. *business men*, *children at school*), meaning the model may struggle with age groups presented in different contexts; for example, a child riding a bicycle or an adult in casual clothing. We will broaden our data collection strategy to cover a wider range of real-world scenarios aligned with our deployment purpose.

**3. No quantitative evaluation metric**
Due to the absence of ground truth labels for test images, model performance was assessed purely through visual inspection of the patch-colored outputs. For the final submission, we plan to either manually label the test set or explore alternative evaluation strategies to compute a concrete accuracy or error rate.


## Future Work

- Expand dataset to 1000+ images per class with diverse contexts
- Introduce labeled test data for quantitative evaluation (accuracy, F1-score)
- Experiment with larger backbones (EfficientNet-B3, ResNet50) as data grows
- Explore sliding-window inference for higher spatial resolution in crowd scenes


## Acknowledgements

All images used in this project were sourced from [Pexels](https://www.pexels.com/) under their free-to-use license.
