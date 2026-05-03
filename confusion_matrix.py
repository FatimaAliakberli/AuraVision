import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from train_model import build_model, build_test_loader, DEVICE, TRAIN_DIR, TEST_DIR

def generate_matrix():
    # 1. Setup Data and Model
    # We need the training mapping to ensure indices match
    from torchvision import datasets
    train_dataset = datasets.ImageFolder(TRAIN_DIR)
    class_to_idx = train_dataset.class_to_idx
    class_names = train_dataset.classes

    test_loader, _ = build_test_loader(TEST_DIR, class_to_idx)
    
    model = build_model()
    model.load_state_dict(torch.load("age_classifier.pth", map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    # 2. Collect Predictions
    print("Gathering predictions for matrix...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 3. Create Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # 4. Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix: Age Group Classifier')
    plt.show()

    # Print detailed metrics
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    generate_matrix()