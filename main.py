import os
from PIL import Image

# Use the FULL absolute path to where the dataset was downloaded
kaggle_dir = r"C:\Users\jebin\.gemini\antigravity\scratch\HelmetDetection\datasets\kaggle_helmet"
images_dir = os.path.join(kaggle_dir, "images")
annotations_dir = os.path.join(kaggle_dir, "annotations")

# List all images
images = os.listdir(images_dir)
print(f"Total Kaggle images: {len(images)}")  # Should print 764

# Open one image
img = Image.open(os.path.join(images_dir, images[0]))
img.show()
