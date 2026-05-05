"""
evaluation_script.py
====================
Runs per-image inference on the test set and prints a full table of
predictions, true labels, confidence scores, and pass/fail status.

A summary of misclassified files is printed at the end as an action list.
"""

import os
import torch
from torchvision import datasets

from train_model import build_model, build_test_loader, DEVICE, TRAIN_DIR, TEST_DIR

# ── Configuration ──────────────────────────────────────────────────────────────
ORIGINAL_MODEL_PATH = "models/age_classifier.pth"
FINETUNED_MODEL_PATH = "models/age_classifier_finetuned.pth"

# ──────────────────────────────────────────────────────────────────────────────


def analyze_individual_files(model_path) -> None:
    """Loads the model and prints per-image results with a failure summary."""

    # Establish canonical class mapping from the training directory
    train_dataset = datasets.ImageFolder(TRAIN_DIR)
    class_to_idx  = train_dataset.class_to_idx
    class_names   = train_dataset.classes

    test_loader, test_dataset = build_test_loader(TEST_DIR, class_to_idx)

    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print(f"Analyzing {len(test_dataset)} test images...\n")

    results = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            probs  = torch.softmax(model(inputs), dim=1)
            confidences, preds = torch.max(probs, 1)

            preds       = preds.cpu().numpy()
            labels      = labels.numpy()
            confidences = confidences.cpu().numpy()

            for j in range(len(preds)):
                dataset_idx = i * test_loader.batch_size + j
                img_path, _ = test_dataset.samples[dataset_idx]
                results.append({
                    "filename":   os.path.basename(img_path),
                    "true_class": class_names[labels[j]],
                    "pred_class": class_names[preds[j]],
                    "confidence": confidences[j],
                    "is_correct": labels[j] == preds[j],
                })

    # Results table
    print(f"{'FILENAME':<35} | {'TRUE':<12} | {'PREDICTION':<12} | CONFIDENCE")
    print("-" * 80)
    wrong_count = 0
    for res in results:
        mark = "✓" if res["is_correct"] else "✗"
        if not res["is_correct"]:
            wrong_count += 1
        print(f"{res['filename']:<35} | {res['true_class']:<12} | "
              f"{res['pred_class']:<12} | {res['confidence']:.2%} {mark}")

    # Failure summary
    print("\n" + "=" * 40)
    print(f"TOTAL WRONG: {wrong_count} / {len(results)}")
    print("=" * 40)

    if wrong_count > 0:
        print("\nMISCLASSIFIED FILES (Action Items):")
        for res in results:
            if not res["is_correct"]:
                print(f"  - {res['filename']}: "
                      f"was {res['true_class']}, "
                      f"predicted {res['pred_class']} "
                      f"({res['confidence']:.2%})")


if __name__ == "__main__":
    analyze_individual_files(ORIGINAL_MODEL_PATH)
    analyze_individual_files(FINETUNED_MODEL_PATH)
