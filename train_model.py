import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np

# Step 1. Configurations & Settings

# Paths of our dataset
TRAIN_DIR   = "dataset/train"   # subfolders: children/, adults/, seniors/
TEST_DIR    = "dataset/test"    
MODEL_SAVE  = "age_classifier.pth"

# Training Hyperparameters
NUM_EPOCHS      = 40
BATCH_SIZE      = 8             # small because dataset is tiny
LEARNING_RATE   = 1e-4
IMAGE_SIZE      = 224
NUM_CLASSES     = 3             # children, adults, seniors
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["adults", "children", "seniors"]  


# Step 2. Data Processing 

# Training transforms:
# these create slightly different versions of the same images
# to help the model learn better from a small dataset, data augmentation 
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

def build_dataloaders(train_dir):
    full_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)

    global CLASS_NAMES
    CLASS_NAMES = full_dataset.classes
    print(f"Classes detected: {CLASS_NAMES}")

    # Split 85% train / 15% validation
    n_total  = len(full_dataset)
    n_val    = max(1, int(0.15 * n_total))
    n_train  = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply lighter transforms for validation data
    val_set.dataset = copy.deepcopy(full_dataset)
    val_set.dataset.transform = val_transforms

    # Handle class imbalance by giving higher chance to underrepresented classes
    targets      = [full_dataset.targets[i] for i in train_set.indices]
    class_counts = np.bincount(targets)
    print(f"Train class distribution: { {CLASS_NAMES[i]: int(class_counts[i]) for i in range(len(CLASS_NAMES))} }")
    
    weights      = 1.0 / class_counts
    sample_weights = [weights[t] for t in targets]
    sampler      = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


# Step 3. The Model
def build_model():
    # Load a pretrained EfficientNet-B0 model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze all pretrained layers first: 
    # We don't need to re-learn basic shapes and colors
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last part of the feature extractor:
    # so the model can adapt better to this task  
    for param in model.features[6:].parameters():
        param.requires_grad = True

    # Replace the final classifier with a new one for 3 classes
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

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        # Training Phase:
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

        # Validation Phase:
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

# Step 5. Inference 
def run_inference(model, test_dir):
    # Switch model to evaluation mode
    model.eval()

    supported = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith(supported)
    ])

    if not image_files:
        print(f"No images found in {test_dir}")
        return

    print("\n" + "=" * 65)
    print(f"{'Image':<35} {'Prediction':<12} {'Confidence':>10}")
    print("=" * 65)

    results = []
    for fname in image_files:
        img_path = os.path.join(test_dir, fname)
        try:
            img    = Image.open(img_path).convert("RGB")
            tensor = val_transforms(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

            pred_idx    = int(np.argmax(probs))
            pred_class  = CLASS_NAMES[pred_idx]
            confidence  = probs[pred_idx]

            all_probs = {CLASS_NAMES[i]: f"{probs[i]:.2%}" for i in range(NUM_CLASSES)}

            print(f"{fname:<35} {pred_class:<12} {confidence:>9.2%}")
            print(f"↳ All scores: {all_probs}")

            results.append({
                "image":      fname,
                "prediction": pred_class,
                "confidence": confidence,
                "scores":     all_probs,
            })

        except Exception as e:
            print(f"{fname:<35} ERROR: {e}")

    print("=" * 65)

    # Summary
    from collections import Counter
    preds = [r["prediction"] for r in results]
    counts = Counter(preds)
    print(f"\n📊 Prediction summary across {len(results)} test images:")
    for cls, cnt in counts.most_common():
        print(f"   {cls}: {cnt} image(s)")

    return results

# Step 6. Execution
def main():
    print(f" Running on: {DEVICE}\n")

    print(" Loading training data...")
    train_loader, val_loader = build_dataloaders(TRAIN_DIR)

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

    print(f"\n Running inference on test images in '{TEST_DIR}'...")
    run_inference(model, TEST_DIR)


if __name__ == "__main__":
    main()