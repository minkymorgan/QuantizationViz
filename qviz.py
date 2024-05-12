import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def quantize(image, bits):
    max_val = 2**bits - 1
    quantized = np.floor(image / 256 * max_val) / max_val * 255
    return quantized.astype(np.uint8)

def display_quantization(image):
    img_array = np.array(image, dtype=np.float32)  # Convert to 32-bit float

    # Adjusting the figure size and layout here
    fig, axes = plt.subplots(5, 1, figsize=(5, 15))  # 5 rows, 1 column, larger figure size
    axes[0].imshow(img_array / 255)  # Display original image
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    for i, bits in enumerate([8, 4, 2, 1]):
        quantized_img = quantize(img_array, bits)
        axes[i + 1].imshow(quantized_img)
        axes[i + 1].set_title(f'{bits}-bit Quantized')
        axes[i + 1].axis('off')

    plt.tight_layout()
    st.pyplot(fig)  # Display the plot in Streamlit

st.title('Image Quantization Demo')

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    display_quantization(image)

