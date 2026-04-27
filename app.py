import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

st.title("Brain Tumor Classification")

uploaded = st.file_uploader("Upload une image MRI", type=["jpg", "png"])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption="Image uploadée", width=300)

    if st.button("Prédire"):
        # placeholder
        st.warning("Modèle pas encore chargé")