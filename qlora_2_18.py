import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import matplotlib.pyplot as plt

# Title and Description
st.title("Understanding QLoRA: Quantization and Low-Rank Adaptation")

st.write("""
### What is QLoRA?
QLoRA stands for Quantized Low-Rank Adaptation. It's a technique used in training large language models (LLMs) that combines quantization and low-rank adaptation. This approach significantly reduces computational requirements and costs associated with training large AI models.

### Why is QLoRA Important?
Training large language models like GPT-3 or GPT-4 requires substantial computational resources. QLoRA changes the economics of AI by making it feasible to "top up" the training of pre-trained models efficiently and cost-effectively.

### Visualization
Upload two images to visualize the concept. One image represents the pre-trained model, and the other represents new data to be learned.
""")

# Define default images
default_image1_path = "screenshots/SanFranciso_GoldenGateBridge.jpeg"
default_image2_path = "screenshots/StoneHenge.jpeg"

# Load default images
default_image1 = Image.open(default_image1_path).convert('RGB')
default_image2 = Image.open(default_image2_path).convert('RGB')

# File uploaders for the images
uploaded_file1 = st.file_uploader("Choose the first image (pre-trained model, e.g., Golden Gate Bridge)...", type=["jpg", "jpeg", "png"])
uploaded_file2 = st.file_uploader("Choose the second image (new data, e.g., Stonehenge)...", type=["jpg", "jpeg", "png"])

# Use uploaded images if provided, otherwise default images
if uploaded_file1 is not None:
    image1 = Image.open(uploaded_file1).convert('RGB')
else:
    image1 = default_image1

if uploaded_file2 is not None:
    image2 = Image.open(uploaded_file2).convert('RGB')
else:
    image2 = default_image2

# Ensure images are the same size
image2 = image2.resize(image1.size)

# Sidebar options for user to select quantization bit depth and adaptation factor
bit_depth = st.sidebar.slider("Select Quantization Bit Depth", min_value=2, max_value=8, value=4, step=1)
adapt_factor = st.sidebar.slider("Select Adaptation Enhancement Factor", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
rank = st.sidebar.slider("Select Rank for Low-Rank Approximation", min_value=1, max_value=100, value=20, step=1)

# Caching function to store results using st.cache_data
@st.cache_data
def process_images(image1, image2, bit_depth, adapt_factor, rank):
    # Function to quantize an image to a lower bit-depth
    def quantize_image(image, bit_depth):
        image_array = np.array(image)
        scale = (2 ** bit_depth - 1) / 255.0
        quantized_array = (image_array * scale).astype(np.uint8)
        dequantized_array = (quantized_array / scale).astype(np.uint8)
        quantized_image = Image.fromarray(dequantized_array)
        return quantized_image

    # Function to perform low-rank approximation
    def low_rank_approximation(image, rank):
        channels = []
        for i in range(3):  # For R, G, B channels
            U, s, Vt = np.linalg.svd(image[:, :, i], full_matrices=False)
            S = np.diag(s)
            low_rank_channel = np.dot(U[:, :rank], np.dot(S[:rank, :rank], Vt[:rank, :]))
            channels.append(low_rank_channel)
        low_rank_image = np.stack(channels, axis=2)
        return np.clip(low_rank_image, 0, 255).astype(np.uint8)

    # Function to simulate low-rank adaptation by selectively enhancing parts of the image
    def adapt_image(image, pattern_image, factor=1.5):
        grayscale_pattern = ImageOps.grayscale(pattern_image)
        edges = grayscale_pattern.filter(ImageFilter.FIND_EDGES)
        mask = edges.point(lambda x: 255 if x > 30 else 0).convert("1")
        enhancer = ImageEnhance.Contrast(image)
        enhanced_image = enhancer.enhance(factor)
        enhanced_image = enhanced_image.resize(image.size).convert("RGB")
        image = image.resize(image.size).convert("RGB")
        mask = mask.resize(image.size).convert("L")
        adapted_image = Image.composite(enhanced_image, image, mask)
        return adapted_image

    quantized_image1 = quantize_image(image1, bit_depth)
    quantized_image2 = quantize_image(image2, bit_depth)
    low_rank_image1 = low_rank_approximation(np.array(quantized_image1), rank)
    low_rank_image2 = low_rank_approximation(np.array(quantized_image2), rank)
    adapted_image1 = adapt_image(quantized_image1, quantized_image2, adapt_factor)
    return quantized_image1, quantized_image2, low_rank_image1, low_rank_image2, adapted_image1

# Process the images
quantized_image1, quantized_image2, low_rank_image1, low_rank_image2, adapted_image1 = process_images(image1, image2, bit_depth, adapt_factor, rank)

# Function to plot the color histogram of an image
def plot_color_histogram(image, ax):
    image_array = np.array(image)
    colors = ['r', 'g', 'b']
    for i, color in enumerate(colors):
        hist, bins = np.histogram(image_array[:, :, i], bins=256, range=(0, 256))
        ax.plot(bins[:-1], hist, color=color)
    ax.set_xlim([0, 256])
    ax.set_yticks([])

# Create the layout with four columns and histograms
fig, axes = plt.subplots(4, 4, figsize=(20, 20))

# Row 1 - Original and Quantized Images with Histograms
axes[0, 0].imshow(image1)
axes[0, 0].set_title('Original First Image')
axes[0, 0].axis('off')
plot_color_histogram(image1, axes[0, 1])
axes[0, 1].set_title('Histogram Original First Image')
axes[0, 2].imshow(quantized_image1)
axes[0, 2].set_title('Quantized First Image')
axes[0, 2].axis('off')
plot_color_histogram(quantized_image1, axes[0, 3])
axes[0, 3].set_title('Histogram Quantized First Image')

st.write(""" 
Your LLM (represented by your photo) is quantised to compress it, reduce cost.
""")

# Row 2 - Original and Quantized Second Images with Histograms
axes[1, 0].imshow(image2)
axes[1, 0].set_title('Original Second Image')
axes[1, 0].axis('off')
plot_color_histogram(image2, axes[1, 1])
axes[1, 1].set_title('Histogram Original Second Image')
axes[1, 2].imshow(quantized_image2)
axes[1, 2].set_title('Quantized Second Image')
axes[1, 2].axis('off')
plot_color_histogram(quantized_image2, axes[1, 3])
axes[1, 3].set_title('Histogram Quantized Second Image')

st.write("""
Your private finetuning data is prepared similarly.
""")

# New Row - Low-Rank Approximations of the Second Image (Update)
axes[2, 0].imshow(low_rank_image2[:, :, 0], cmap='Reds', alpha=0.5)
axes[2, 0].set_title('Update Red Channel')
axes[2, 0].axis('off')
axes[2, 1].imshow(low_rank_image2[:, :, 1], cmap='Greens', alpha=0.5)
axes[2, 1].set_title('Update Green Channel')
axes[2, 1].axis('off')
axes[2, 2].imshow(low_rank_image2[:, :, 2], cmap='Blues', alpha=0.5)
axes[2, 2].set_title('Update Blue Channel')
axes[2, 2].axis('off')
plot_color_histogram(low_rank_image2, axes[2, 3])
axes[2, 3].set_title('Histogram Update Channels')

st.write("""
Above, we process the finetuning update, also selecting low-rank approximations (here on each colour, with SVD), and an average update calculated.
""")

# Row 3 - Low-Rank Approximations and Final Adapted Image
axes[3, 0].imshow(low_rank_image1[:, :, 0], cmap='Reds')
axes[3, 0].set_title('Low-Rank Red Channel')
axes[3, 0].axis('off')
axes[3, 1].imshow(low_rank_image1[:, :, 1], cmap='Greens')
axes[3, 1].set_title('Low-Rank Green Channel')
axes[3, 1].axis('off')
axes[3, 2].imshow(low_rank_image1[:, :, 2], cmap='Blues')
axes[3, 2].set_title('Low-Rank Blue Channel')
axes[3, 2].axis('off')
axes[3, 3].imshow(adapted_image1)
axes[3, 3].set_title('Adapted Image')
axes[3, 3].axis('off')

st.write("""
Low rank partitioning of the LLM is done (here on each colour, with SVD), and an average weight calculated. Training done to correct it using the private finetuning data.
""")

st.pyplot(fig)

# Display Final Adapted Image and its Histogram
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))
axes2[0].imshow(adapted_image1)
axes2[0].set_title('Final Adapted Image')
axes2[0].axis('off')
plot_color_histogram(adapted_image1, axes2[1])
axes2[1].set_title('Histogram Adapted Image')

plt.tight_layout()
st.pyplot(fig2)

st.write("""

Finally the compressed LLM is adapted by the low rank updates.

### Conclusion

The low-rank approximation technique demonstrates how we can reduce the complexity of data while preserving its essential features. In the context of QLoRA, this technique is used to fine-tune large language models efficiently by focusing on the most important parameters.

Thank you for exploring QuantizationViz. We hope this tool enhances your understanding of quantization and low-rank adaptation in both digital imaging and artificial intelligence model compression.
""")

