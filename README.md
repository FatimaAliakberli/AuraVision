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


> **Crowd Age Group Majority Classifier** — Milestone 2

## Table of Contributions
 
| Team Member | Contribution |
| :--- | :--- |
| Fatima Alakbarli | Model Training & Fine-tuning |
| Shahd Elaydy | Data Collection & Evaluation |
 
---
 
## What's New in Milestone 2
 
Milestone 2 represents a significant expansion of the project across every dimension: data, model, and evaluation.
 
**Dataset improvements:** The dataset was substantially enlarged and diversified. Search queries were broadened to cover a wider range of real-world contexts — for example, children are now represented not just in school settings but also in playgrounds, youth sports, and birthday parties; adults appear in airports, gyms, festivals, and tech conferences; seniors are captured in gardens, tour groups, and community centres. The result is a far more representative dataset that reduces context-specific bias from Milestone 1.
 
**Model improvements:** The base EfficientNet-B0 model now goes through a two-stage pipeline — initial training followed by dedicated fine-tuning. The fine-tuning stage applies stronger regularisation (heavier dropout, higher weight decay, cosine annealing with warm restarts, and label smoothing) specifically to reduce the overfitting that was observed in Milestone 1. Test-time augmentation (TTA) is also used during the final accuracy estimate to produce a more reliable result.
 
**Evaluation improvements:** Milestone 1 had no quantitative evaluation due to unlabelled test data. Milestone 2 introduces a fully labelled test set and two dedicated evaluation tools: a per-image inference table with confidence scores and a confusion matrix with a classification report.
 
---
 
## Dataset
 
Images were collected from [Pexels](https://www.pexels.com/) using the Pexels API, with significantly expanded and diversified search queries compared to Milestone 1.
 
### Training Queries
 
| Class | Search Queries |
|---|---|
| **Children** | `children`, `children at school`, `playground children`, `children playing`, `kindergarten`, `youth sports` |
| **Adults** | `business women`, `business men`, `university students`, `new married couples`, `adults working`, `music festival adults`, `adults in gym`, `airport terminal` |
| **Seniors** | `seniors`, `seniors walking`, `retirement community`, `gardening seniors`, `seniors socializing` |
 
### Test Queries (held-out, context-varied)
 
| Class | Search Queries |
|---|---|
| **Children** | `Elementary school assembly`, `Family day at zoo`, `Kids birthday party background`, `Youth soccer match crowd` |
| **Adults** | `City commuters morning`, `Airport terminal crowd`, `Tech conference audience`, `Nightlife street scene`, `Busy food court` |
| **Seniors** | `Public park morning walkers`, `Senior center ballroom`, `Older people gardening together`, `Pensioners traveling in tour group` |
 
---
 
## Project Structure
 
```
AuraVision/
│
├── dataset/                        ← Download from Google Drive (see below)
│   ├── train/
│   │   ├── Children/
│   │   ├── Adults/
│   │   └── Seniors/
│   └── test/
│       ├── Children/
│       ├── Adults/
│       └── Seniors/
│
├── extracted_data/                 ← Raw downloaded images (before train/test split)
│
├── models/
│   ├── age_classifier.pth          ← Base trained model
│   └── age_classifier_finetuned.pth ← Fine-tuned model (best for inference)
│
├── src/
│   ├── data_extraction.py          ← Downloads images from Pexels API
│   ├── split_data.py               ← Splits extracted_data/ into dataset/
│   ├── train_model.py              ← EfficientNet-B0 training logic
│   ├── finetune_model.py           ← Fine-tuning with stronger regularisation
│   ├── evaluation_script.py        ← Per-image inference table
│   └── confusion_matrix.py         ← Confusion matrix + classification report
│
├── 01_Data_Extraction_and_Split.ipynb
├── 02_Model_Training.ipynb
├── 03_Evaluation_and_Confusion_Matrix.ipynb
├── 04_Finetuning.ipynb
│
└── README.md
```
 
---
 
## Model Architecture
 
AuraVision uses **transfer learning** on top of **EfficientNet-B0** pretrained on ImageNet.
 
```
EfficientNet-B0 (pretrained on ImageNet)
  ├── Features [0–5]  → Frozen (general visual features)
  └── Features [6+]   → Trainable (task-specific adaptation)
           ↓
  Custom Classifier Head
  ├── Dropout (0.4)
  ├── Linear → 128
  ├── ReLU
  ├── Dropout (0.3)
  └── Linear → 3 classes (Adults / Children / Seniors)
```
 
### Base Training Strategy
 
- Heavy data augmentation (random crops, flips, rotation, colour jitter) to compensate for dataset size
- Weighted random sampling to handle class imbalance
- AdamW optimiser with cosine annealing scheduler
- 40 epochs, batch size 8, learning rate 1e-4
### Fine-Tuning Strategy
 
After the base model is trained, `finetune_model.py` runs a dedicated fine-tuning pass to reduce overfitting:
 
- **Backbone fully frozen** — only the classifier head is retrained, preventing the backbone from re-memorising the training data
- **Heavier regularisation** — dropout increased to 0.5/0.4 (from 0.4/0.3), weight decay increased to 1e-2 (from 1e-4)
- **Label smoothing** (0.1) to prevent the model from becoming overconfident
- **Cosine annealing with warm restarts** (`T_0=10`) to help the optimiser escape local minima during the shorter fine-tuning run
- **Weighted random sampler** to maintain class balance during fine-tuning
- **Test-time augmentation (TTA)** — at evaluation time, 5 different augmented versions of each image are passed through the model and their logits are averaged, producing a more robust accuracy estimate than a single forward pass
- 20 fine-tuning epochs, learning rate 1e-4
Fine-tuning produces `models/age_classifier_finetuned.pth`, which is the recommended model for inference.
 
---
 
## How to Use
 
### Option A — Use the Pre-built Dataset (Recommended)
 
If you want to skip data collection and go straight to training or evaluation, download the ready-made dataset from Google Drive:
 
**[Download dataset/ folder](https://drive.google.com/drive/folders/17IctSoWxnCqxz8_tLTxxqlZT6A0YK620?usp=sharing)**
 
After downloading, place the `dataset/` folder in the root of your project directory so that the structure matches the layout shown above.
 
> **Just want to see results without re-training?** Run `03_Evaluation_and_Confusion_Matrix.ipynb` directly. It loads the saved model checkpoint and produces the per-image results table and confusion matrix immediately — no training time required.
 
---
 
### Option B — Run the Full Pipeline from Scratch
 
Follow the notebooks in order:
 
#### Step 1 — Data Extraction (`01_Data_Extraction_and_Split.ipynb`)
 
> **Requires a Pexels API key.** Sign up for free at [pexels.com/api](https://www.pexels.com/api/) and add your key to a `.env` file in the project root:
> ```
> PEXELS_API_KEY=your_actual_key_here
> ```
 
This notebook downloads images from Pexels into `extracted_data/` (training images) and `test_dataset/` (test images), then splits the training images into `dataset/train/` and `dataset/test/` using an 80/20 ratio.
 
> **Want to inspect the raw downloaded images before the split?** Browse the pre-split extracted images here:
> **[View extracted_data/ on Google Drive](https://drive.google.com/drive/folders/1jP8pLFAvmEIdhhla0pxQqnDpmdlmv76u?usp=sharing)**
 
```bash
pip install requests python-dotenv
```
 
Then run all cells in `01_Data_Extraction_and_Split.ipynb`.
 
#### Step 2 — Model Training (`02_Model_Training.ipynb`)
 
Trains EfficientNet-B0 on the split dataset for 40 epochs and saves the best checkpoint to `models/age_classifier.pth`.
 
```bash
pip install torch torchvision pillow numpy
```
 
Then run all cells in `02_Model_Training.ipynb`.
 
#### Step 3 — Fine-Tuning (`04_Finetuning.ipynb`)
 
Loads `age_classifier.pth`, applies the stronger regularisation described above, and saves the improved model to `models/age_classifier_finetuned.pth`.
 
Run all cells in `04_Finetuning.ipynb`.
 
#### Step 4 — Evaluation (`03_Evaluation_and_Confusion_Matrix.ipynb`)
 
Runs inference on the test set and produces:
- A per-image table showing the true label, predicted label, confidence score, and pass/fail status for every image
- A summary of all misclassified files
- A confusion matrix heatmap
- A full classification report (precision, recall, F1-score per class)
Both the base model and the fine-tuned model are evaluated side-by-side so the improvement from fine-tuning is visible.
 
Run all cells in `03_Evaluation_and_Confusion_Matrix.ipynb`.
 
---
 
### Installing All Dependencies
 
```bash
pip install torch torchvision pillow numpy scikit-learn seaborn matplotlib requests python-dotenv
```
 
---
 
## Evaluation
 
Milestone 2 introduces proper quantitative evaluation with a fully labelled test set.
 
### Per-Image Results
 
`evaluation_script.py` (called from Notebook 03) prints a table for every test image:
 
```
FILENAME                            | TRUE         | PREDICTION   | CONFIDENCE
--------------------------------------------------------------------------------
kids_birthday_party_0.jpg           | Children     | Children     | 94.21% ✓
city_commuters_morning_3.jpg        | Adults       | Seniors      | 61.08% ✗
...
 
TOTAL WRONG: X / N
```
 
Misclassified files are listed at the end as an action list for further data collection or inspection.
 
### Confusion Matrix
 
`confusion_matrix.py` generates a heatmap showing how predictions are distributed across the three classes, and prints a full `sklearn` classification report with per-class precision, recall, and F1-score.
 
---
 
## Limitations & Future Work
 
- Larger backbones such as EfficientNet-B3 or ResNet-50 could be explored as data grows
- Sliding-window or patch-based inference at higher resolution for dense crowd scenes
- Deployment as a real-time inference API for integration into venue management systems
---
 
## Acknowledgements
 
All images used in this project were sourced from [Pexels](https://www.pexels.com/) under their free-to-use licence.
