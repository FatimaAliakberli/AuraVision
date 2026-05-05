import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets

from train_model import build_model, build_test_loader, DEVICE, TRAIN_DIR, TEST_DIR

# Configuration:
ORIGINAL_MODEL_PATH = "models/age_classifier.pth"
FINETUNED_MODEL_PATH = "models/age_classifier_finetuned.pth"

def generate_matrix(model_path: str) -> None:
    # Loads a checkpoint, runs inference on the test set, and plots the matrix

    # Use training directory to establish the canonical class mapping
    train_dataset = datasets.ImageFolder(TRAIN_DIR)
    class_to_idx  = train_dataset.class_to_idx
    class_names   = train_dataset.classes

    test_loader, _ = build_test_loader(TEST_DIR, class_to_idx)

    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_preds  = []
    all_labels = []

    print("Gathering predictions...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs  = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — Age Group Classifier")
    plt.tight_layout()
    plt.show()

    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


if __name__ == "__main__":
    generate_matrix(ORIGINAL_MODEL_PATH)
    generate_matrix(FINETUNED_MODEL_PATH)
