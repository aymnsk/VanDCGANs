import streamlit as st
import pickle
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

model.eval()

st.title("🎨 AI Style Transfer or Image Generator")

uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    output_image = transforms.ToPILImage()(output.squeeze(0).clamp(0, 1))
    st.image(output_image, caption='Generated Output', use_column_width=True)
