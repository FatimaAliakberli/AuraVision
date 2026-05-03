<<<<<<< HEAD
"""
Age Group Majority Classifier
==============================
Trains a transfer learning model (EfficientNet-B0) on three classes:
  - children
  - adults
  - seniors

Then runs inference on a structured test set and reports per-class
accuracy and a full classification report.

Expected folder structure:
    split_dataset/
        train/
            children/   *.jpg / *.png ...
            adults/     *.jpg / *.png ...
            seniors/    *.jpg / *.png ...
        test/
            children/   *.jpg / *.png ...
            adults/     *.jpg / *.png ...
            seniors/    *.jpg / *.png ...

Usage:
    python age_group_classifier.py
"""

=======
>>>>>>> 379f409d413be5eedc32be4f3d045ab9c4851ff9
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from collections import Counter

<<<<<<< HEAD
# ─────────────────────────────────────────────
# CONFIG  — adjust paths here
# ─────────────────────────────────────────────
TRAIN_DIR   = "split_dataset/train"   # subfolders: children/, adults/, seniors/
TEST_DIR    = "test_dataset"    # same subfolder structure as train
=======
# Step 1. Configurations & Settings

# Paths of our dataset
TRAIN_DIR   = "dataset/train"   # subfolders: children/, adults/, seniors/
TEST_DIR    = "dataset/test"    
>>>>>>> 379f409d413be5eedc32be4f3d045ab9c4851ff9
MODEL_SAVE  = "age_classifier.pth"

# Training Hyperparameters
NUM_EPOCHS      = 40
BATCH_SIZE      = 8
LEARNING_RATE   = 1e-4
IMAGE_SIZE      = 224
NUM_CLASSES     = 3             # children, adults, seniors
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

<<<<<<< HEAD
# Will be set after ImageFolder scans the train directory.
# Keeping it here as a reference; never derive it from the test folder.
CLASS_NAMES = []

# ─────────────────────────────────────────────
# 1. DATA TRANSFORMS
# ─────────────────────────────────────────────
=======
CLASS_NAMES = ["adults", "children", "seniors"]  


# Step 2. Data Processing 

# Training transforms:
# these create slightly different versions of the same images
# to help the model learn better from a small dataset, data augmentation 
>>>>>>> 379f409d413be5eedc32be4f3d045ab9c4851ff9
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Transforms for validation: 
val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

<<<<<<< HEAD
# ─────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────
=======
>>>>>>> 379f409d413be5eedc32be4f3d045ab9c4851ff9
def build_dataloaders(train_dir):
    """
    Scans train_dir with ImageFolder to discover classes.
    The class-to-index mapping produced here is the canonical one
    used for ALL subsequent operations, including test evaluation.
    """
    global CLASS_NAMES

    full_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)

<<<<<<< HEAD
    # Store the canonical class list from the TRAINING directory only.
=======
    global CLASS_NAMES
>>>>>>> 379f409d413be5eedc32be4f3d045ab9c4851ff9
    CLASS_NAMES = full_dataset.classes
    print(f"Classes detected from train dir: {CLASS_NAMES}")
    print(f"Class-to-index mapping:          {full_dataset.class_to_idx}")

    # 85 / 15 split
    n_total = len(full_dataset)
    n_val   = max(1, int(0.15 * n_total))
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

<<<<<<< HEAD
    # Lighter transforms for the validation split
=======
    # Apply lighter transforms for validation data
>>>>>>> 379f409d413be5eedc32be4f3d045ab9c4851ff9
    val_set.dataset = copy.deepcopy(full_dataset)
    val_set.dataset.transform = val_transforms

    # Handle class imbalance by giving higher chance to underrepresented classes
    targets      = [full_dataset.targets[i] for i in train_set.indices]
    class_counts = np.bincount(targets)
    print(f"Train class distribution: { {CLASS_NAMES[i]: int(class_counts[i]) for i in range(len(CLASS_NAMES))} }")
<<<<<<< HEAD
    weights        = 1.0 / class_counts
=======
    
    weights      = 1.0 / class_counts
>>>>>>> 379f409d413be5eedc32be4f3d045ab9c4851ff9
    sample_weights = [weights[t] for t in targets]
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, full_dataset.class_to_idx


def build_test_loader(test_dir, train_class_to_idx):
    """
    Loads the test set using the SAME class-to-index mapping that was
    established during training.  Even if ImageFolder would normally
    produce a different alphabetical order, we force it to match training.

    Returns the DataLoader and the underlying ImageFolder dataset.
    """
    # Build a temporary ImageFolder just to discover which files exist
    # in which subfolder — we do NOT use its class_to_idx.
    raw_test = datasets.ImageFolder(test_dir, transform=val_transforms)

    # Remap every sample's label from the test folder's own index to the
    # training index.  This is the key step that prevents a mismatch.
    test_class_to_idx = raw_test.class_to_idx   # may differ from training!
    print(f"\nTest dir class-to-index (raw):    {test_class_to_idx}")
    print(f"Train class-to-index (canonical): {train_class_to_idx}")

    # Build a remapping table: test_label → train_label
    remap = {}
    for class_name, test_idx in test_class_to_idx.items():
        if class_name not in train_class_to_idx:
            raise ValueError(
                f"Class '{class_name}' found in test set but not in training set. "
                f"Training classes: {list(train_class_to_idx.keys())}"
            )
        remap[test_idx] = train_class_to_idx[class_name]

    # Apply the remapping in-place
    raw_test.samples = [(path, remap[label]) for path, label in raw_test.samples]
    raw_test.targets = [remap[label] for label in raw_test.targets]
    # Also fix class_to_idx and classes so they match training
    raw_test.class_to_idx = train_class_to_idx
    raw_test.classes = list(CLASS_NAMES)

    test_loader = DataLoader(raw_test, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader, raw_test

<<<<<<< HEAD
# ─────────────────────────────────────────────
# 3. MODEL
# ─────────────────────────────────────────────
=======

# Step 3. The Model
>>>>>>> 379f409d413be5eedc32be4f3d045ab9c4851ff9
def build_model():
    # Load a pretrained EfficientNet-B0 model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

<<<<<<< HEAD
    for param in model.parameters():
        param.requires_grad = False

    for param in model.features[6:].parameters():
        param.requires_grad = True

=======
    # Freeze all pretrained layers first: 
    # We don't need to re-learn basic shapes and colors
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last part of the feature extractor:
    # so the model can adapt better to this task  
    for param in model.features[6:].parameters():
        param.requires_grad = True

    # Replace the final classifier with a new one for 3 classes
>>>>>>> 379f409d413be5eedc32be4f3d045ab9c4851ff9
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, NUM_CLASSES),
    )

    return model.to(DEVICE)

# Step 4. Training the Model 
def train_model(model, train_loader, val_loader):
<<<<<<< HEAD
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
=======
    # Loss function for multi-class classification
    criterion  = nn.CrossEntropyLoss()
    
    # Optimizer updates only the trainable parameters
    optimizer  = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=1e-4
    )
    
    # Learning rate schedular:
    # Automatically slows down learning speed as we get closer to the best result
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
>>>>>>> 379f409d413be5eedc32be4f3d045ab9c4851ff9

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
<<<<<<< HEAD
        # Train
=======
        # Training Phase:
>>>>>>> 379f409d413be5eedc32be4f3d045ab9c4851ff9
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds      = torch.max(outputs, 1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

<<<<<<< HEAD
        # Validate
=======
        # Validation Phase:
>>>>>>> 379f409d413be5eedc32be4f3d045ab9c4851ff9
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs        = model(inputs)
                _, preds       = torch.max(outputs, 1)
                val_correct   += (preds == labels).sum().item()
                val_total     += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        scheduler.step()

        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2%}  "
              f"Val Acc: {val_acc:.2%}")

        # Save the best model based on validation accuracy
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    print(f"\n Best validation accuracy: {best_val_acc:.2%}")
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), MODEL_SAVE)
    print(f" Model saved to: {MODEL_SAVE}")
    return model

<<<<<<< HEAD
# ─────────────────────────────────────────────
# 5. INFERENCE ON STRUCTURED TEST SET
# ─────────────────────────────────────────────
def run_inference(model, test_loader, test_dataset):
    """
    Runs inference on a structured test set (subfolders per class).
    Reports per-image predictions AND overall / per-class accuracy.
    Ground-truth labels come from the folder structure, remapped to
    the canonical training class indices — so there is no label leakage.
    """
=======
# Step 5. Inference 
def run_inference(model, test_dir):
    # Switch model to evaluation mode
>>>>>>> 379f409d413be5eedc32be4f3d045ab9c4851ff9
    model.eval()

    all_preds  = []
    all_labels = []
    all_probs  = []

<<<<<<< HEAD
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = np.argmax(probs, axis=1)
=======
    if not image_files:
        print(f"No images found in {test_dir}")
        return
>>>>>>> 379f409d413be5eedc32be4f3d045ab9c4851ff9

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

    # ── Per-image table ──────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"{'Image':<35} {'True':<12} {'Predicted':<12} {'Confidence':>10}")
    print("=" * 75)

    image_paths = [s[0] for s in test_dataset.samples]

    for i, (path, true_idx, pred_idx, probs) in enumerate(
            zip(image_paths, all_labels, all_preds, all_probs)):
        fname      = os.path.basename(path)
        true_cls   = CLASS_NAMES[true_idx]
        pred_cls   = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx]
        match_mark = "✓" if true_idx == pred_idx else "✗"

<<<<<<< HEAD
        score_str = {CLASS_NAMES[j]: f"{probs[j]:.2%}" for j in range(NUM_CLASSES)}
        print(f"{fname:<35} {true_cls:<12} {pred_cls:<12} {confidence:>9.2%}  {match_mark}")
        print(f"  ↳ All scores: {score_str}")

    print("=" * 75)
=======
            all_probs = {CLASS_NAMES[i]: f"{probs[i]:.2%}" for i in range(NUM_CLASSES)}

            print(f"{fname:<35} {pred_class:<12} {confidence:>9.2%}")
            print(f"↳ All scores: {all_probs}")
>>>>>>> 379f409d413be5eedc32be4f3d045ab9c4851ff9

    # ── Overall accuracy ─────────────────────────────────────────────
    all_preds_arr  = np.array(all_preds)
    all_labels_arr = np.array(all_labels)
    overall_acc    = (all_preds_arr == all_labels_arr).mean()

    print(f"\n📊 Overall test accuracy: {overall_acc:.2%}  ({int((all_preds_arr == all_labels_arr).sum())}/{len(all_labels_arr)})")

    # ── Per-class accuracy ───────────────────────────────────────────
    print("\n📋 Per-class results:")
    print(f"  {'Class':<12} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    print(f"  {'-'*40}")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask      = all_labels_arr == cls_idx
        if mask.sum() == 0:
            print(f"  {cls_name:<12} {'—':>8} {'0':>7} {'—':>10}")
            continue
        correct   = (all_preds_arr[mask] == cls_idx).sum()
        total     = mask.sum()
        acc       = correct / total
        print(f"  {cls_name:<12} {correct:>8} {total:>7} {acc:>9.2%}")

    # ── Prediction distribution ──────────────────────────────────────
    print("\n📊 Prediction distribution across test images:")
    pred_counts = Counter([CLASS_NAMES[p] for p in all_preds])
    for cls, cnt in pred_counts.most_common():
        print(f"   {cls}: {cnt} image(s)")

    return all_preds, all_labels

# Step 6. Execution
def main():
    print(f" Running on: {DEVICE}\n")

<<<<<<< HEAD
    # ── Train ────────────────────────────────────────────────────────
    print("📂 Loading training data...")
    train_loader, val_loader, train_class_to_idx = build_dataloaders(TRAIN_DIR)
=======
    print(" Loading training data...")
    train_loader, val_loader = build_dataloaders(TRAIN_DIR)
>>>>>>> 379f409d413be5eedc32be4f3d045ab9c4851ff9

    print("\n Building model (EfficientNet-B0)...")
    model = build_model()

    # Print model size information
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params:     {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")

    # Train the model
    print(f"\n Training for {NUM_EPOCHS} epochs...")
    model = train_model(model, train_loader, val_loader)

<<<<<<< HEAD
    # ── Test ─────────────────────────────────────────────────────────
    # Pass train_class_to_idx so the test loader uses the SAME label
    # mapping — the test folder's own alphabetical order is discarded.
    print(f"\n📂 Loading test data from '{TEST_DIR}'...")
    test_loader, test_dataset = build_test_loader(TEST_DIR, train_class_to_idx)
    print(f"   Test images found: {len(test_dataset)}")

    print(f"\n🔍 Running inference on test set...")
    run_inference(model, test_loader, test_dataset)
=======
    print(f"\n Running inference on test images in '{TEST_DIR}'...")
    run_inference(model, TEST_DIR)
>>>>>>> 379f409d413be5eedc32be4f3d045ab9c4851ff9


if __name__ == "__main__":
    main()