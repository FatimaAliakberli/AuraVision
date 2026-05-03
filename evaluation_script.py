import torch
import numpy as np
import os
from torchvision import datasets
from train_model import build_model, build_test_loader, DEVICE, TRAIN_DIR, TEST_DIR

def analyze_individual_files():
    # 1. Setup Data and Model
    train_dataset = datasets.ImageFolder(TRAIN_DIR)
    class_to_idx = train_dataset.class_to_idx
    class_names = train_dataset.classes

    # Load test data and get the dataset object specifically to access .samples
    test_loader, test_dataset = build_test_loader(TEST_DIR, class_to_idx)
    
    model = build_model()
    model.load_state_dict(torch.load("age_classifier.pth", map_location=DEVICE))
    model.eval()

    # 2. Run Inference
    print(f"Analyzing {len(test_dataset)} test images...\n")
    
    results = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, 1)
            
            # Get data for each image in the current batch
            preds = preds.cpu().numpy()
            labels = labels.numpy()
            confidences = confidences.cpu().numpy()
            
            for j in range(len(preds)):
                # Index in the overall dataset
                dataset_idx = i * test_loader.batch_size + j
                img_path, _ = test_dataset.samples[dataset_idx]
                filename = os.path.basename(img_path)
                
                results.append({
                    "filename": filename,
                    "true_class": class_names[labels[j]],
                    "pred_class": class_names[preds[j]],
                    "confidence": confidences[j],
                    "is_correct": labels[j] == preds[j]
                })

    # 3. Print Results
    print(f"{'FILENAME':<35} | {'TRUE CLASS':<12} | {'PREDICTION':<12} | {'CONFIDENCE'}")
    print("-" * 85)
    
    wrong_count = 0
    for res in results:
        status = "✓" if res['is_correct'] else "✗"
        if not res['is_correct']:
            wrong_count += 1
            
        print(f"{res['filename']:<35} | {res['true_class']:<12} | {res['pred_class']:<12} | {res['confidence']:.2%} {status}")

    # 4. Summary of Failures
    print("\n" + "="*30)
    print(f"TOTAL WRONG: {wrong_count} / {len(results)}")
    print("="*30)
    
    if wrong_count > 0:
        print("\nTOP MISCLASSIFIED FILES (Action Items):")
        for res in results:
            if not res['is_correct']:
                print(f" - {res['filename']}: Was {res['true_class']}, model thought {res['pred_class']} ({res['confidence']:.2%})")

if __name__ == "__main__":
    analyze_individual_files()