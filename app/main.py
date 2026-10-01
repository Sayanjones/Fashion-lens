import streamlit as st
from PIL import Image

from classifier import CLASS_NAMES, load_model, predict, preprocess

st.set_page_config(page_title="Fashion MNIST classifier", page_icon="👕")


@st.cache_resource
def get_model():
    # Cached so the model loads once per server process, not on every click.
    return load_model()


st.title("Fashion item classifier")
st.write(
    "Upload a picture of a clothing item. A small CNN trained on Fashion MNIST "
    "will guess which of 10 categories it belongs to."
)

with st.sidebar:
    st.header("Input options")
    invert = st.radio(
        "Invert colors",
        options=["auto", "yes", "no"],
        help=(
            "Fashion MNIST items are light on a dark background. Photos with a "
            "white background need to be inverted. 'auto' decides from the "
            "brightness of the image border."
        ),
    )
    st.caption("Classes: " + ", ".join(CLASS_NAMES))

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    x = preprocess(image, invert=invert)

    left, right = st.columns(2)
    with left:
        st.subheader("Your image")
        st.image(image.resize((160, 160)))
    with right:
        st.subheader("What the model sees")
        st.image(x[0, :, :, 0], width=160, clamp=True)
        st.caption("28x28 grayscale, after resizing and optional inversion")

    if st.button("Classify"):
        label, confidence, probs = predict(get_model(), x)
        st.success(f"Prediction: **{label}** ({confidence:.1%} confidence)")

        top3 = probs.argsort()[::-1][:3]
        st.bar_chart({CLASS_NAMES[i]: float(probs[i]) for i in top3})
        st.caption(
            "Confidence is the softmax of the model's raw outputs. It is not a "
            "calibrated probability, and it will happily be high on images that "
            "look nothing like the training data."
        )
