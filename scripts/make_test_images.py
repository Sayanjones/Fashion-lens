"""Save the first test-set image of each class as a PNG in test_images/.

Run from the repo root:  python scripts/make_test_images.py
"""
import os

import numpy as np
from PIL import Image
from tensorflow import keras

CLASS_NAMES = [
    "tshirt_top", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle_boot",
]

(_, _), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
os.makedirs("test_images", exist_ok=True)

for class_id, name in enumerate(CLASS_NAMES):
    idx = int(np.where(y_test == class_id)[0][0])
    img = Image.fromarray(X_test[idx]).resize((224, 224), Image.Resampling.BICUBIC)
    path = f"test_images/{class_id}_{name}.png"
    img.save(path)
    print(path, f"(test index {idx})")
