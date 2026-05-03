import requests
import os

# --- CONFIGURATION ---
API_KEY = '45fFNqCjN4SWAHfgFxsRGTqrx1sZVoml5EFSpOdX0y9Utxyf96cujulx'
SAVE_DIR = 'crowd_dataset'
CLASSES = {
    "Children": ["children", "children at school", "playground children", "children playing", "kindergarten", "youth sports", ],
    "Adults": ["business women", "business men", "university students", "new married couples", "adults working", "music festival adults", "adults in gym", "airport terminal"],
    "Seniors": ["seniors", "seniors walking", "retirement community", "gardening seniors", "seniors socializing"]
}
IMAGES_PER_QUERY = 40 # Adjust based on your needs

def download_images():
    headers = {'Authorization': API_KEY}
    
    for label, queries in CLASSES.items():
        # Create folder for the class
        class_path = os.path.join(SAVE_DIR, label)
        os.makedirs(class_path, exist_ok=True)
        
        for query in queries:
            url = f"https://api.pexels.com/v1/search?query={query}&per_page={IMAGES_PER_QUERY}"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                for i, photo in enumerate(data['photos']):
                    img_url = photo['src']['large'] # Use 'large' or 'original'
                    img_data = requests.get(img_url).content
                    
                    # Save image
                    filename = f"{query.replace(' ', '_')}_{i}.jpg"
                    with open(os.path.join(class_path, filename), 'wb') as f:
                        f.write(img_data)
                print(f"Finished downloading '{query}' for class '{label}'")
            else:
                print(f"Error {response.status_code} for query {query}")

if __name__ == "__main__":
    download_images()