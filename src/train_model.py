"""
train_model.py
==============
Transfer-learning trainer for AuraVision (EfficientNet-B0, 3 classes).

Shared constants and helper functions are imported by:
  - confusion_matrix.py
  - evaluation_script.py
  - finetune_model.py

Run standalone to produce models/age_classifier.pth.
"""

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

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_DIR   = "dataset/train"   # subfolders: Children/, Adults/, Seniors/
TEST_DIR    = "dataset/test"    # same subfolder structure as train
MODEL_SAVE  = "AuraVision/models/age_classifier.pth"

NUM_EPOCHS    = 40
BATCH_SIZE    = 8
LEARNING_RATE = 1e-4
IMAGE_SIZE    = 224
NUM_CLASSES   = 3
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set after ImageFolder scans the train directory — do NOT derive from test dir.
CLASS_NAMES: list[str] = []
# ──────────────────────────────────────────────────────────────────────────────


# ── Data Transforms ────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Data Loaders ───────────────────────────────────────────────────────────────
def build_dataloaders(train_dir: str):
    """
    Scans train_dir with ImageFolder to discover classes.
    The class-to-index mapping produced here is the canonical one
    used for ALL subsequent operations, including test evaluation.

    Returns: train_loader, val_loader, class_to_idx
    """
    global CLASS_NAMES

    full_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    CLASS_NAMES  = full_dataset.classes

    print(f"Classes detected:     {CLASS_NAMES}")
    print(f"Class-to-index map:   {full_dataset.class_to_idx}")

    # 85 / 15 train-validation split
    n_total = len(full_dataset)
    n_val   = max(1, int(0.15 * n_total))
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Use lighter transforms for the validation split
    val_set.dataset           = copy.deepcopy(full_dataset)
    val_set.dataset.transform = val_transforms

    # Weighted sampler to handle class imbalance
    targets        = [full_dataset.targets[i] for i in train_set.indices]
    class_counts   = np.bincount(targets)
    sample_weights = [1.0 / class_counts[t] for t in targets]
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights))

    print(f"Train class dist: { {CLASS_NAMES[i]: int(class_counts[i]) for i in range(len(CLASS_NAMES))} }")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, full_dataset.class_to_idx


def build_test_loader(test_dir: str, train_class_to_idx: dict):
    """
    Loads the test set using the SAME class-to-index mapping established
    during training, preventing label mismatches across runs.

    Returns: test_loader, test_dataset
    """
    raw_test = datasets.ImageFolder(test_dir, transform=val_transforms)

    # Remap test labels to match the training index
    remap = {}
    for class_name, test_idx in raw_test.class_to_idx.items():
        if class_name not in train_class_to_idx:
            raise ValueError(
                f"Class '{class_name}' found in test set but not in training set. "
                f"Training classes: {list(train_class_to_idx.keys())}"
            )
        remap[test_idx] = train_class_to_idx[class_name]

    raw_test.samples    = [(path, remap[label]) for path, label in raw_test.samples]
    raw_test.targets    = [remap[label] for label in raw_test.targets]
    raw_test.class_to_idx = train_class_to_idx
    raw_test.classes    = list(CLASS_NAMES)

    test_loader = DataLoader(raw_test, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader, raw_test


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model() -> nn.Module:
    """
    EfficientNet-B0 pretrained on ImageNet.
    Backbone layers 0-5 frozen; layers 6+ and the custom head are trainable.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze later feature blocks so they can adapt to this task
    for param in model.features[6:].parameters():
        param.requires_grad = True

    # Replace the classifier head for 3 classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, NUM_CLASSES),
    )

    return model.to(DEVICE)


# ── Training Loop ─────────────────────────────────────────────────────────────
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
    """Trains for NUM_EPOCHS and saves the best checkpoint by validation accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        # ── Train phase ──────────────────────────────────────────────
        model.train()
        running_loss = correct = total = 0
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

        # ── Validation phase ─────────────────────────────────────────
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                _, preds       = torch.max(model(inputs), 1)
                val_correct   += (preds == labels).sum().item()
                val_total     += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        scheduler.step()

        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}]  "
              f"Loss: {train_loss:.4f}  Train: {train_acc:.2%}  Val: {val_acc:.2%}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    print(f"\n✅ Best validation accuracy: {best_val_acc:.2%}")
    os.makedirs("models", exist_ok=True)
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), MODEL_SAVE)
    print(f"✅ Model saved → {MODEL_SAVE}")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model: nn.Module, test_loader: DataLoader, test_dataset) -> tuple:
    """
    Runs inference on the structured test set.
    Prints per-image predictions and a per-class accuracy summary.
    Returns (all_preds, all_labels).
    """
    model.eval()
    all_preds = all_labels = []
    all_probs_list = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            probs  = torch.softmax(model(inputs), dim=1).cpu().numpy()
            preds  = np.argmax(probs, axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs_list.extend(probs.tolist())

    # Per-image table
    print("\n" + "=" * 75)
    print(f"{'Image':<35} {'True':<12} {'Predicted':<12} {'Confidence':>10}")
    print("=" * 75)
    image_paths = [s[0] for s in test_dataset.samples]
    for path, true_idx, pred_idx, probs in zip(image_paths, all_labels, all_preds, all_probs_list):
        fname      = os.path.basename(path)
        mark       = "✓" if true_idx == pred_idx else "✗"
        score_str  = {CLASS_NAMES[j]: f"{probs[j]:.2%}" for j in range(NUM_CLASSES)}
        print(f"{fname:<35} {CLASS_NAMES[true_idx]:<12} {CLASS_NAMES[pred_idx]:<12} "
              f"{probs[pred_idx]:>9.2%}  {mark}")
        print(f"  ↳ All scores: {score_str}")
    print("=" * 75)

    # Overall and per-class accuracy
    preds_arr  = np.array(all_preds)
    labels_arr = np.array(all_labels)
    overall    = (preds_arr == labels_arr).mean()
    print(f"\n📊 Overall test accuracy: {overall:.2%}  ({int((preds_arr == labels_arr).sum())}/{len(labels_arr)})")

    print("\n📋 Per-class results:")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = labels_arr == cls_idx
        if not mask.any():
            continue
        acc = (preds_arr[mask] == cls_idx).mean()
        print(f"  {cls_name:<12} {int((preds_arr[mask]==cls_idx).sum()):>4}/{int(mask.sum()):<4} {acc:.2%}")

    return all_preds, all_labels


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"🖥️  Device: {DEVICE}\n")

    print("📂 Loading training data...")
    train_loader, val_loader, class_to_idx = build_dataloaders(TRAIN_DIR)

    print("\n🏗️  Building model (EfficientNet-B0)...")
    model = build_model()
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params:     {total:,}")
    print(f"   Trainable params: {trainable:,}")

    print(f"\n🚀 Training for {NUM_EPOCHS} epochs...")
    model = train_model(model, train_loader, val_loader)

    print(f"\n📂 Loading test data from '{TEST_DIR}'...")
    test_loader, test_dataset = build_test_loader(TEST_DIR, class_to_idx)
    print(f"   Test images: {len(test_dataset)}")

    print("\n🔍 Running inference on test set...")
    run_inference(model, test_loader, test_dataset)


if __name__ == "__main__":
    main()
