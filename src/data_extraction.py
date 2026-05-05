import os
import requests
from dotenv import load_dotenv

# Load API key from .env 
load_dotenv()
API_KEY = os.getenv("PEXELS_API_KEY")

if not API_KEY:
    raise EnvironmentError(
        "PEXELS_API_KEY not set. Please add it to your .env file:\n"
        "  PEXELS_API_KEY=your_api_key_here"
    )

# Configuration:
SAVE_DIR         = "extracted_data"
IMAGES_PER_QUERY = 40

# Training queries per class
TRAIN_CLASSES = {
    "Children": [
        "children",
        "children at school",
        "playground children",
        "children playing",
        "kindergarten",
        "youth sports",
    ],
    "Adults": [
        "business women",
        "business men",
        "university students",
        "new married couples",
        "adults working",
        "music festival adults",
        "adults in gym",
        "airport terminal",
    ],
    "Seniors": [
        "seniors",
        "seniors walking",
        "retirement community",
        "gardening seniors",
        "seniors socializing",
    ],
}

# Test queries per class (diverse, context-varied)
TEST_CLASSES = {
    "Children": [
        "Elementary school assembly",
        "Family day at zoo",
        "Kids birthday party background",
        "Youth soccer match crowd",
    ],
    "Adults": [
        "City commuters morning",
        "Airport terminal crowd",
        "Tech conference audience",
        "Nightlife street scene",
        "Busy food court",
    ],
    "Seniors": [
        "Public park morning walkers",
        "Senior center ballroom",
        "Older people gardening together",
        "Pensioners traveling in tour group",
    ],
}

def _download_for_classes(classes: dict, save_dir: str) -> None:
    # Downloads images for a given classes dict into save_dir/<ClassName>/
    headers = {"Authorization": API_KEY}

    for label, queries in classes.items():
        class_path = os.path.join(save_dir, label)
        os.makedirs(class_path, exist_ok=True)

        for query in queries:
            url      = f"https://api.pexels.com/v1/search?query={query}&per_page={IMAGES_PER_QUERY}"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                for i, photo in enumerate(data["photos"]):
                    img_url  = photo["src"]["large"]
                    filename = f"{query.replace(' ', '_')}_{i}.jpg"
                    save_path = os.path.join(class_path, filename)

                    # Skip images that were already downloaded
                    if os.path.exists(save_path):
                        continue

                    try:
                        img_response = requests.get(img_url, timeout=20)
                        img_response.raise_for_status()

                        with open(save_path, "wb") as f:
                            f.write(img_response.content)

                    except requests.exceptions.RequestException as e:
                        print(f"Skipped image {i} for query '{query}' because of connection error: {e}")
                        continue
                print(f" '{query}' → {len(data['photos'])} images saved to '{label}/'")
            else:
                print(f" Error {response.status_code} for query '{query}'")


def download_train_images() -> None:
    # Download training images into crowd_dataset/
    print(f"\n Downloading TRAINING images → {SAVE_DIR}/\n")
    _download_for_classes(TRAIN_CLASSES, SAVE_DIR)
    _print_summary(SAVE_DIR)


def download_test_images(test_dir: str = "test_dataset") -> None:
    # Download test images into test_dataset/
    print(f"\n Downloading TEST images → {test_dir}/\n")
    _download_for_classes(TEST_CLASSES, test_dir)
    _print_summary(test_dir)


def _print_summary(base_dir: str) -> None:
    print(f"\n Summary: {base_dir}")
    for cls in os.listdir(base_dir):
        cls_path = os.path.join(base_dir, cls)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path) if f.endswith(".jpg")])
            print(f"  {cls}: {count} images")


if __name__ == "__main__":
    download_train_images()
    download_test_images()
