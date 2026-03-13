"""
Age Group Patch Visualizer
===========================
Uses the trained EfficientNet-B0 classifier to classify patches
of each test image, then overlays colored regions:

    🟢 GREEN  → children
    🔵 BLUE   → adults
    🔴 RED    → seniors

Saves annotated images to: results/

Usage:
    python visualize_results.py

Requirements:
    - age_classifier.pth must exist (run age_group_classifier.py first)
    - dataset/test/  folder with test images
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TEST_DIR        = "dataset/test"
RESULTS_DIR     = "results_updated"
MODEL_PATH      = "age_classifier.pth"
IMAGE_SIZE      = 224
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Grid: how many patches across width and height
# 4x4 = 16 patches per image (good balance of detail vs speed)
GRID_COLS       = 4
GRID_ROWS       = 4

# Transparency of the color overlay (0=invisible, 255=solid)
OVERLAY_ALPHA   = 110

# Class names must match training order (alphabetical = ImageFolder default)
CLASS_NAMES     = ["adults", "children", "seniors"]

# Colors per class: (R, G, B)
CLASS_COLORS = {
    "adults":   (66,  133, 244),   # Blue
    "children": (52,  168,  83),   # Green
    "seniors":  (234,  67,  53),   # Red
}

# ─────────────────────────────────────────────
# TRANSFORMS  (same as val_transforms in training)
# ─────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model(model_path):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, 3),
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"✅ Model loaded from: {model_path}")
    return model

# ─────────────────────────────────────────────
# CLASSIFY A SINGLE PATCH
# ─────────────────────────────────────────────
def classify_patch(model, patch_img):
    """Returns (class_name, confidence, all_probs)"""
    tensor = infer_transform(patch_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx   = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs

# ─────────────────────────────────────────────
# PROCESS ONE IMAGE
# ─────────────────────────────────────────────
def process_image(model, img_path, save_path):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    patch_w = W // GRID_COLS
    patch_h = H // GRID_ROWS

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    class_vote_counts = {cls: 0 for cls in CLASS_NAMES}

    try:
        patch_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except:
        patch_font = ImageFont.load_default()

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x0 = col * patch_w
            y0 = row * patch_h
            x1 = x0 + patch_w if col < GRID_COLS - 1 else W
            y1 = y0 + patch_h if row < GRID_ROWS - 1 else H

            patch        = img.crop((x0, y0, x1, y1))
            cls, conf, _ = classify_patch(model, patch)
            class_vote_counts[cls] += 1

            color = CLASS_COLORS[cls] + (OVERLAY_ALPHA,)
            draw.rectangle([x0, y0, x1, y1], fill=color)

            label = f"{cls[:3].upper()} {conf:.0%}"
            draw.text((x0 + 4, y0 + 4), label, fill=(255, 255, 255, 230), font=patch_font)
            draw.rectangle([x0, y0, x1 - 1, y1 - 1],
                           outline=(255, 255, 255, 160), width=1)

    majority_class = max(class_vote_counts, key=class_vote_counts.get)
    majority_count = class_vote_counts[majority_class]

    img_rgba   = img.convert("RGBA")
    composited = Image.alpha_composite(img_rgba, overlay)
    result_img = composited.convert("RGB")
    result_draw = ImageDraw.Draw(result_img)

    try:
        title_font  = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        legend_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font  = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        title_font = legend_font = small_font = ImageFont.load_default()

    # ── Top banner with majority result ──
    banner_color = CLASS_COLORS[majority_class]
    result_draw.rectangle([0, 0, W, 40], fill=banner_color)
    vote_summary = "  |  ".join(
        [f"{cls.capitalize()}: {class_vote_counts[cls]}" for cls in CLASS_NAMES]
    )
    result_draw.text(
        (10, 9),
        f"MAJORITY: {majority_class.upper()}  ({majority_count}/{GRID_COLS*GRID_ROWS} patches)   [{vote_summary}]",
        fill=(255, 255, 255),
        font=title_font,
    )

    # ── Bottom legend box ──
    lx, ly = 10, H - 95
    result_draw.rectangle([lx - 6, ly - 6, lx + 185, ly + 88],
                          fill=(20, 20, 20))
    result_draw.text((lx, ly), "Legend", fill=(255, 255, 255), font=legend_font)
    ly += 24
    for cls, color in CLASS_COLORS.items():
        result_draw.rectangle([lx, ly, lx + 16, ly + 16], fill=color)
        result_draw.text((lx + 22, ly), cls.capitalize(),
                         fill=(255, 255, 255), font=small_font)
        ly += 22

    result_img.save(save_path, quality=95)
    return majority_class, class_vote_counts

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"🖥️  Device: {DEVICE}")
    model = load_model(MODEL_PATH)

    supported   = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = sorted([
        f for f in os.listdir(TEST_DIR)
        if f.lower().endswith(supported)
    ])

    if not image_files:
        print(f"⚠️  No images found in {TEST_DIR}")
        return

    print(f"\n🖼️  Processing {len(image_files)} images with {GRID_ROWS}×{GRID_COLS} patch grid...\n")
    print(f"{'Image':<50} {'Majority':<12} Patch Votes")
    print("─" * 80)

    for fname in image_files:
        img_path  = os.path.join(TEST_DIR, fname)
        save_name = os.path.splitext(fname)[0] + "_result.jpg"
        save_path = os.path.join(RESULTS_DIR, save_name)

        try:
            majority, votes = process_image(model, img_path, save_path)
            vote_str = "  ".join([f"{c}: {v}" for c, v in votes.items()])
            print(f"{fname:<50} {majority:<12} {vote_str}")
        except Exception as e:
            print(f"{fname:<50} ERROR: {e}")

    print(f"\n✅ All results saved to: {RESULTS_DIR}/")
    print(f"   🔵 Blue = Adults  |  🟢 Green = Children  |  🔴 Red = Seniors")

if __name__ == "__main__":
    main()