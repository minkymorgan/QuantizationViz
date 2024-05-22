import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import matplotlib.pyplot as plt
import os

# Title and Description
st.title("Understanding QLoRA: Quantization and Low-Rank Adaptation")
st.write("""
### What is QLoRA?
QLoRA stands for Quantized Low-Rank Adaptation. It's a technique used in training large language models (LLMs) that combines two powerful methods: quantization and low-rank adaptation. This approach significantly reduces the computational requirements and costs associated with training large AI models.

### Why is QLoRA Important?
Training large language models like GPT-3 or GPT-4 requires substantial computational resources, often running into millions of dollars. For many organizations, especially those with limited GPU budgets, this is prohibitively expensive. QLORA changes the economics of AI by making it feasible to "top up" the training of pre-trained models efficiently and cost-effectively.

### How Does QLoRA Work?
1. **Quantization**: Reduces the precision of the model's parameters, lowering the memory and computational requirements. For example, converting 32-bit floating point numbers to 8-bit integers.
2. **Low-Rank Adaptation**: Instead of updating all parameters, it updates only a small, low-rank subset of them. This targeted update maintains performance while minimizing computational costs.

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

# Function to quantize an image to a lower bit-depth
def quantize_image(image, bit_depth):
    image_array = np.array(image)
    scale = (2 ** bit_depth - 1) / 255.0
    quantized_array = (image_array * scale).astype(np.uint8)
    dequantized_array = (quantized_array / scale).astype(np.uint8)
    quantized_image = Image.fromarray(dequantized_array)
    return quantized_image

# Function to simulate low-rank adaptation by selectively enhancing parts of the image
def adapt_image(image, pattern_image, factor=1.5):
    # Convert the pattern image to grayscale and find edges
    grayscale_pattern = ImageOps.grayscale(pattern_image)
    edges = grayscale_pattern.filter(ImageFilter.FIND_EDGES)

    # Create a mask from the edges
    mask = edges.point(lambda x: 255 if x > 30 else 0).convert("1")

    # Apply selective enhancement based on the mask
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(factor)

    # Ensure all images have the same size and mode
    enhanced_image = enhanced_image.resize(image.size).convert("RGB")
    image = image.resize(image.size).convert("RGB")
    mask = mask.resize(image.size).convert("L")

    # Blend the enhanced image with the original using the mask
    adapted_image = Image.composite(enhanced_image, image, mask)
    return adapted_image

# Function to plot the color histogram of an image
def plot_color_histogram(image, ax):
    image_array = np.array(image)
    colors = ['r', 'g', 'b']
    for i, color in enumerate(colors):
        hist, bins = np.histogram(image_array[:, :, i], bins=256, range=(0, 256))
        ax.plot(bins[:-1], hist, color=color)
    ax.set_xlim([0, 256])
    ax.set_yticks([])

# Quantize both images to the selected bit depth
quantized_image1 = quantize_image(image1, bit_depth)
quantized_image2 = quantize_image(image2, bit_depth)

# Adapt the quantized images using patterns from the other image
adapted_image1 = adapt_image(quantized_image1, quantized_image2, adapt_factor)

# Create the layout with four columns and histograms
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Row 1
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

# Row 2
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

plt.tight_layout()
st.pyplot(fig)

# Display Adapted Image and its Histogram
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))

axes2[0].imshow(adapted_image1)
axes2[0].set_title('Adapted Image 1')
axes2[0].axis('off')

plot_color_histogram(adapted_image1, axes2[1])
axes2[1].set_title('Histogram Adapted Image 1')

plt.tight_layout()
st.pyplot(fig2)

st.write("""
### Conclusion

The QLORA technique means that you can download a base model costing millions of dollars to train, and tailor it to train on your specefic data to "top it up" so that it better performs on your use case, whilst still retaining it's original value, and you can do so affordably.

But, as I explained previously, quantisation can reduce quality and is lossy, so you need to carefully experiment on how to get this right for your use cases. Another technique called Retrieval Augmented Generation helps to counteract the hullucination in quantised LLM outputs, by crafting a information rich prompt that retrieves the relevant (and now missing) documents from a vector database) and this is often used with QLORA.

# Last Remarks

The AI community is driving innovations at a breakneck pace and techniques like Quantisation are production ready now, and new techniques like KANs are already being discussed as the next innovation to alter AI economics.

If you would like assistance or guidence - email me at andrew.morgan(at)6point6.co.uk

Further Reading

QLoRA, which stands for Quantized Low-Rank Adaptation, was introduced as a method to efficiently finetune large language models (LLMs). This technique was developed by researchers including Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer from the University of Washington's NLP group. The primary goal of QLoRA is to reduce the memory and computational requirements for finetuning LLMs without sacrificing performance.

QLoRA was officially presented in 2023 and has been discussed in the context of NeurIPS 2023. The method involves several key innovations:

1. **4-bit NormalFloat (NF4) Quantization**: This is an information-theoretically optimal data type for normally distributed weights, reducing the memory footprint while maintaining precision.
2. **Double Quantization**: This technique quantizes the quantization constants themselves, further reducing memory usage.
3. **Paged Optimizers**: These manage memory spikes during gradient checkpointing, preventing out-of-memory errors and facilitating the use of large models on single GPUs.

QLoRA achieves high-fidelity 4-bit finetuning by backpropagating gradients through a frozen, 4-bit quantized pretrained model into Low Rank Adapters (LoRA). This allows finetuning of models with up to 65 billion parameters on a single 48GB GPU, maintaining performance equivalent to full 16-bit precision finetuning.

The practical implications of QLoRA are significant. By reducing the memory requirements, QLoRA makes it possible for organizations with limited GPU resources to finetune and utilize large language models effectively. This democratizes access to advanced AI capabilities, enabling more widespread use and customization of LLMs in various applications.

For further reading and detailed technical information, you can refer to the sources:
- [GitHub - QLoRA](https://github.com/artidoro/qlora)
- [NeurIPS 2023 Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)
- [arXiv - QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Papers With Code - QLoRA](https://paperswithcode.com/paper/qlora-efficient-finetuning-of-quantized-llms)

""")

