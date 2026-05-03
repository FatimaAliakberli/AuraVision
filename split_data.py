import os
import shutil
import random

def split_dataset(source_dir, output_dir, split_ratio=0.8):
    # Set up the new folder structure
    for folder in ['train', 'test']:
        for class_name in os.listdir(source_dir):
            os.makedirs(os.path.join(output_dir, folder, class_name), exist_ok=True)

    classes = os.listdir(source_dir)
    
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        # Get all images and shuffle them
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(images)
        
        # Calculate split index
        split_point = int(len(images) * split_ratio)
        train_images = images[:split_point]
        test_images = images[split_point:]
        
        # Helper to copy files
        def copy_files(files, subset):
            for f in files:
                src = os.path.join(class_path, f)
                dst = os.path.join(output_dir, subset, class_name, f)
                shutil.copy(src, dst)
        
        copy_files(train_images, 'train')
        copy_files(test_images, 'test')
        
        print(f"Class '{class_name}': {len(train_images)} train, {len(test_images)} test.")

# --- EXECUTION ---
# source_dir is your original 'crowd_dataset'
# output_dir is where the new split folders will be created
split_dataset(source_dir='crowd_dataset', output_dir='split_dataset')