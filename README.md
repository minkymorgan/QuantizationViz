# QuantizationViz: Image Quantization Demonstration

Welcome to QuantizationViz, an interactive web application designed to visually demonstrate the effects of image quantization. This tool helps users understand how quantization impacts image quality and AI model performance.

## Project Overview

QuantizationViz invites users to upload images and apply different levels of quantization to observe the resulting quality and compression. This interactive demonstration elucidates the core principles of quantization, a critical technique used in optimizing AI models.

In the landscape of AI and machine learning, models are often published with various quantization levels, allowing them to be adapted for different hardware constraints and performance needs. However, the impact of these quantization levels on model performance and memory footprint is not widely understood. By visualizing how quantization affects digital images, this application aims to provide a tangible understanding of these impacts, making it easier for users to appreciate the trade-offs involved in AI model compression.

If you would like to explore these concepts further, a live version of the app is available for you to experiment with:

[QuantizationViz Live Demo](https://minkymorgan-qviz.streamlit.app/)



## App Introduction

The following screenshots provide an overview of the app's purpose, and what it does:

* user uploads image, and this is displayed as uploaded, resized.
* python converts image to numpy array data, and applies quantisation to it
* many different levels of quantisation are applied, degrading the image

The images reveal how the quantisation compresses the parameters in the AI model, by _rounding_ values down from 16 bits, to 8, 4, 2 and 1 bit.
While this quantisation process can speeding up computation, some sacrifices in quality should be expected.


![App Introduction_1](./screenshots/screenshot_1.png)
![App Introduction_2](./screenshots/screenshot_2.png)
![App Introduction_3](./screenshots/screenshot_3.png)
## Conclusion

QuantizationViz is designed to educate users about the balance between compression and quality in digital imaging and AI, providing an intuitive understanding through direct interaction and visualization.

Thank you for exploring QuantizationViz. We hope this tool enhances your understanding of quantization effects in both digital imaging and artificial intelligence model compression.

Here is a small gallery for you to review:

![Sample Gallery](./screenshots/AI_UnderstandingQuantizationThroughViz.png)



