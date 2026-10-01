import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import classifier  # noqa: E402


def make_image(mode="RGB", size=(300, 400), color=0):
    return Image.new(mode, size, color)


def test_preprocess_shape_dtype_range():
    x = classifier.preprocess(make_image(color=(120, 30, 200)), invert="no")
    assert x.shape == (1, 28, 28, 1)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() <= x.max() <= 1.0


def test_preprocess_accepts_rgba_and_grayscale():
    assert classifier.preprocess(make_image("RGBA", color=(0, 0, 0, 0))).shape == (1, 28, 28, 1)
    assert classifier.preprocess(make_image("L", color=128)).shape == (1, 28, 28, 1)


def test_transparent_png_becomes_white_not_black():
    x = classifier.preprocess(make_image("RGBA", color=(0, 0, 0, 0)), invert="no")
    assert x.mean() > 0.99


def test_auto_invert_flips_light_background():
    white_bg = make_image("L", color=255)
    # A dark square in the middle on a white background
    white_bg.paste(0, (100, 100, 300, 300))
    inverted = classifier.preprocess(white_bg, invert="auto")
    untouched = classifier.preprocess(white_bg, invert="no")
    assert inverted[0, 0, 0, 0] < 0.1       # background became dark
    assert untouched[0, 0, 0, 0] > 0.9


def test_auto_invert_leaves_dark_background_alone():
    dark_bg = make_image("L", color=0)
    dark_bg.paste(255, (100, 100, 300, 300))
    x = classifier.preprocess(dark_bg, invert="auto")
    assert x[0, 0, 0, 0] < 0.1


def test_softmax_sums_to_one():
    p = classifier.softmax(np.array([[1000.0, 0.0, -1000.0]]))
    assert np.isclose(p.sum(), 1.0)
    assert not np.isnan(p).any()


@pytest.mark.skipif(not os.path.exists(classifier.MODEL_PATH), reason="trained model not present")
def test_model_end_to_end_smoke():
    model = classifier.load_model()
    x = classifier.preprocess(make_image("L", color=0), invert="no")
    label, conf, probs = classifier.predict(model, x)
    assert label in classifier.CLASS_NAMES
    assert probs.shape == (10,)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)
    assert 0.0 <= conf <= 1.0
