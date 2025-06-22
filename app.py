import streamlit as st
import torch
import matplotlib.pyplot as plt
from generator import Generator

@st.cache_resource
def load_generator():
    gen = Generator()
    gen.load_state_dict(torch.load("generator.pth", map_location="cpu"))
    gen.eval()
    return gen

st.title("🎨 MNIST Handwritten Digit Generator")
generator = load_generator()

digit = st.selectbox("Select a digit to generate (0–9):", range(10))
if st.button("Generate 5 Samples"):
    noise = torch.randn(5, 100)
    labels = torch.full((5,), digit, dtype=torch.long)
    with torch.no_grad():
        imgs = generator(noise, labels).cpu()
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for ax, img in zip(axes, imgs):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.axis("off")
    st.pyplot(fig)
