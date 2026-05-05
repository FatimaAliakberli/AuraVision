# AuraVision

> **Crowd Age Group Majority Classifier**

AuraVision is a computer vision project designed to identify the **dominant age group within a crowd scene**. Rather than relying on facial recognition, the model learns to distinguish age groups through visual cues such as clothing style, accessories, body posture, and contextual belongings — making it more privacy-conscious and robust in real-world scenarios. The ultimate goal is to deploy AuraVision in settings such as public spaces, retail environments, or event venues where understanding the age composition of a crowd can inform decision-making.

## Milestone 1 — Summary
 
In Milestone 1, we built the initial version of AuraVision using a small manually collected dataset of **117 images** across three classes (Children, Adults, Seniors), sourced from Pexels using a narrow set of search queries (e.g. *business men*, *children at school*, *seniors walking*). We trained an EfficientNet-B0 model with transfer learning, freezing the early backbone layers and fine-tuning a custom 3-class head using AdamW with cosine annealing over 40 epochs.
 
The model achieved **over 90% accuracy on the training data**, but without labelled test data we could only evaluate it visually — dividing test images into 4×4 patch grids and inspecting colour-coded predictions. Children were the most reliably identified class (~70% of patches correct); Adults and Seniors were frequently confused due to similar clothing styles. The core limitations were the small dataset size, narrow query diversity, and the complete absence of quantitative evaluation metrics.

## What's New in Milestone 2
 
Milestone 2 represents a significant expansion of the project across every dimension: data, model, and evaluation.
 
**Dataset improvements:** The dataset was substantially enlarged and diversified. Search queries were broadened to cover a wider range of real-world contexts — for example, children are now represented not just in school settings but also in playgrounds, youth sports, and birthday parties; adults appear in airports, gyms, festivals, and tech conferences; seniors are captured in gardens, tour groups, and community centres. The result is a far more representative dataset that reduces context-specific bias from Milestone 1.
 
**Model improvements:** The base EfficientNet-B0 model now goes through a two-stage pipeline — initial training followed by dedicated fine-tuning. The fine-tuning stage applies stronger regularisation (heavier dropout, higher weight decay, cosine annealing with warm restarts, and label smoothing) specifically to reduce the overfitting that was observed in Milestone 1. Test-time augmentation (TTA) is also used during the final accuracy estimate to produce a more reliable result.
 
**Evaluation improvements:** Milestone 1 had no quantitative evaluation due to unlabelled test data. Milestone 2 introduces a fully labelled test set and two dedicated evaluation tools: a per-image inference table with confidence scores and a confusion matrix with a classification report.


## Table of Contributions
 
| Team Member | Contribution |
| :--- | :--- |
| Fatima Alakbarli | Model Training & Fine-tuning |
| Shahd Elaydy | Data Collection & Evaluation |

## Dataset
 
Images were collected from [Pexels](https://www.pexels.com/) using the Pexels API, with significantly expanded and diversified search queries compared to Milestone 1.

### Queries
 
| Class | Search Queries |
|---|---|
| **Children** | `children`, `children at school`, `playground children`, `children playing`, `kindergarten`, `youth sports`, `Elementary school assembly`, `Family day at zoo`, `Kids birthday party background`, `Youth soccer match crowd` |
| **Adults** | `business women`, `business men`, `university students`, `new married couples`, `adults working`, `music festival adults`, `adults in gym`, `airport terminal`, `City commuters morning`, `Airport terminal crowd`, `Tech conference audience`, `Nightlife street scene`, `Busy food court` |
| **Seniors** | `seniors`, `seniors walking`, `retirement community`, `gardening seniors`, `seniors socializing`, `Public park morning walkers`, `Senior center ballroom`, `Older people gardening together`, `Pensioners traveling in tour group` |

 
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
├── extracted_data/                 ← Raw downloaded images(before train/test split) to download from Google Drive see below
|   ├── Children/
│   ├── Adults/
│   └── Seniors/
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
