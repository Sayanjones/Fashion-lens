"""Model loading, preprocessing and prediction for the Fashion MNIST app.

Kept separate from the Streamlit UI so it can be tested without a browser.
"""
import os

import numpy as np
from PIL import Image

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Resolve the path from this file's location, not from the current working
# directory. The same code then works on a laptop, in Docker, anywhere.
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(WORKING_DIR, "trained_model", "trained_fashion_mnist_model.keras")


def load_model(path: str = MODEL_PATH):
    # Imported here so that importing this module stays cheap (and so the
    # preprocessing tests do not need to load TensorFlow at all).
    from tensorflow import keras

    return keras.models.load_model(path)


def looks_like_light_background(gray: np.ndarray) -> bool:
    """True if the image border is mostly bright.

    Fashion MNIST items are light on a dark background. Most real product
    photos are the opposite (dark item, white background), and a model trained
    on the dataset will see those as a different kind of image entirely.
    """
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    return float(border.mean()) > 127.0


def preprocess(image: Image.Image, invert: str = "auto") -> np.ndarray:
    """PIL image -> float32 array of shape (1, 28, 28, 1) with values in [0, 1].

    invert: "auto" (decide from the border brightness), "yes" or "no".
    """
    if image.mode in ("RGBA", "LA", "P"):
        # Flatten transparency onto white instead of letting it turn black.
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)

    gray = image.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype="float32")

    if invert == "yes" or (invert == "auto" and looks_like_light_background(arr)):
        arr = 255.0 - arr

    arr = arr / 255.0  # same scaling as training
    return arr.reshape(1, 28, 28, 1)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def predict(model, x: np.ndarray):
    """Return (class_name, confidence, probabilities[10])."""
    logits = model.predict(x, verbose=0)  # the model outputs logits
    probs = softmax(logits)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs
