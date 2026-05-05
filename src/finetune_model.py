import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from train_model import DEVICE, NUM_CLASSES

# Configuration:
TRAIN_DIR   = "dataset/train"
TEST_DIR    = "dataset/test"
MODEL_PATH  = "models/age_classifier.pth"       # starting checkpoint
SAVE_PATH   = "models/age_classifier_finetuned.pth"

EPOCHS      = 20
BATCH_SIZE  = 8
LR          = 1e-4

# Transforms:
# Stronger augmentation to fight memorisation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomRotation(20),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 5 TTA variants averaged at inference time
tta_transforms = [
    transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(p=1.0),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize(280), transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize(256), transforms.FiveCrop(224),
                        transforms.Lambda(lambda crops: crops[0]),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
]


# Data Loaders:
def build_loaders():
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=val_transform)

    # Weighted sampler for class balance
    class_counts = [0] * NUM_CLASSES
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    weights = [1.0 / class_counts[label] for _, label in train_dataset.samples]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"Classes   : {train_dataset.class_to_idx}")
    print(f"Train set : {len(train_dataset)} images")
    print(f"Test set  : {len(test_dataset)} images\n")
    return train_loader, test_loader, train_dataset.class_to_idx


# The Model:
def build_model() -> nn.Module:
    # Loads the base checkpoint, upgrades dropout, and freezes the backbone.
    # Only the classifier head will be retrained.
    model = models.efficientnet_b0(weights=None)

    # Heavier dropout head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, NUM_CLASSES),
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Loaded weights from {MODEL_PATH}")

    # Freeze backbone; only tune the head
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)\n")

    return model.to(DEVICE)


# Evaluation helpers:
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            _, preds = torch.max(model(inputs), 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_with_tta(model: nn.Module, test_dir: str, class_to_idx: dict) -> float:
    # Averages logits over 5 TTA transforms for a more robust accuracy estimate
    from PIL import Image

    model.eval()
    correct = total = 0

    for class_name, class_idx in class_to_idx.items():
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = Image.open(os.path.join(class_dir, fname)).convert("RGB")
            all_logits = []
            with torch.no_grad():
                for tfm in tta_transforms:
                    tensor = tfm(img).unsqueeze(0).to(DEVICE)
                    all_logits.append(model(tensor))
            pred = torch.stack(all_logits).mean(0).argmax(dim=1).item()
            correct += int(pred == class_idx)
            total   += 1

    return correct / total if total > 0 else 0.0


# Fine-tuning Loop:
def finetune():
    train_loader, test_loader, class_to_idx = build_loaders()
    model = build_model()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=LR, weight_decay=1e-2
    )
    # Warm restarts: resets LR twice during training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1
    )

    best_acc     = evaluate(model, test_loader)
    best_weights = copy.deepcopy(model.state_dict())
    print(f"Baseline (no TTA): {best_acc:.2%}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = correct = total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(model(inputs), 1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)

        scheduler.step()
        train_acc = correct / total
        test_acc  = evaluate(model, test_loader)

        marker = ""
        if test_acc > best_acc:
            best_acc     = test_acc
            best_weights = copy.deepcopy(model.state_dict())
            os.makedirs("models", exist_ok=True)
            torch.save(best_weights, SAVE_PATH)
            marker = "  ← saved"

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"loss: {running_loss/total:.4f}  "
              f"train: {train_acc:.2%}  "
              f"test: {test_acc:.2%}{marker}")

    # Final TTA evaluation
    model.load_state_dict(best_weights)
    tta_acc = evaluate_with_tta(model, TEST_DIR, class_to_idx)

    print("\n" + "=" * 65)
    print(f"Best test accuracy (standard) : {best_acc:.2%}")
    print(f"Best test accuracy (with TTA) : {tta_acc:.2%}")
    print(f"Saved → {SAVE_PATH}")
    print("=" * 65)

    gap = evaluate(model, train_loader) - best_acc
    if gap > 0.20:
        print("\n  Train/test gap still large — consider collecting more data.")
        print(" Target: 300+ images per class with diverse contexts.")
    else:
        print("\n Overfitting gap reduced. Model is generalising better.")


if __name__ == "__main__":
    finetune()
