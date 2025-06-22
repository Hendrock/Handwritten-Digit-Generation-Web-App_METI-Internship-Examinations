import streamlit as st
import torch
import matplotlib.pyplot as plt
from generator import CVAE

@st.cache_resource
def load_decoder():
    model = CVAE()
    model.load_state_dict(torch.load("best_cvae.pth", map_location="cpu"))
    model.eval()
    return model

st.title("🧠 CVAE Handwritten Digit Generator")
decoder = load_decoder()

digit = st.selectbox("Select a digit to generate (0–9):", list(range(10)))
if st.button("Generate 5 Samples"):
    z = torch.randn(5, 20)
    labels = torch.full((5,), digit, dtype=torch.long)
    with torch.no_grad():
        imgs = decoder.decode(z, labels)
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for ax, img in zip(axes, imgs):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.axis("off")
    st.pyplot(fig)
