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
    fig, axes = plt.subplots(5, 1, figsize=(5, 15))  # Setup a vertical subplot layout
    axes[0].imshow(img_array / 255)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    for i, bits in enumerate([8, 4, 2, 1]):
        quantized_img = quantize(img_array, bits)
        axes[i + 1].imshow(quantized_img)
        axes[i + 1].set_title(f'{bits}-bit Quantized')
        axes[i + 1].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

st.title('Image Quantization Demo')

# HTML Description of Quantization
st.markdown("""
<h1>Understanding Quantization in AI</h1>
<p>Quantization is a technique used to reduce the numerical precision of model parameters in machine learning and deep learning. By lowering the bit depth of numbers, quantization helps compress AI models, making them faster and less resource-intensive, thus suitable for deployment on devices with limited computing power.</p>
<p>However, quantization comes with trade-offs. While it significantly reduces the model size and increases inference speed, it can also lead to loss of accuracy and model performance if not handled correctly. It's crucial in AI to find a balance where quantization compresses the model without critically degrading its performance.</p>
<p>This concept of quantization isn't limited to AI models. It's also applicable in digital imaging, where reducing the color depth of images can help understand the balance between compression and quality loss. Below, you can upload an image and see how applying different levels of quantization affects its quality. This visual demonstration helps illustrate the effects quantization might have on more complex systems like neural networks.</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image to see quantization effects", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    display_quantization(image)

