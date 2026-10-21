Overview

This project explores AI-based image style transfer by comparing two deep-learning approaches: CycleGAN and VGG19 Neural Style Transfer. Both methods transform the appearance of the same landscape photograph using different techniques.

CycleGAN translates a photograph into an artistic domain inspired by Monet's paintings, while VGG19 Neural Style Transfer combines the content of the original photograph with the visual style of Vincent van Gogh's The Starry Night.

Objectives

Understand AI-based image style transfer.

Apply a pretrained CycleGAN model for photo-to-Monet translation.

Use VGG19 for neural style transfer.

Preserve the original image's content while applying artistic characteristics.

Compare the outputs of two different generative AI approaches.

Technologies Used

Python

Jupyter Notebook

PyTorch

TorchVision

CycleGAN

VGG19

NumPy

Matplotlib

Pillow (PIL)

Methods Implemented

1. CycleGAN

CycleGAN performs image-to-image domain translation using a pretrained generator. In this experiment, it transforms a landscape photograph into an image inspired by Monet's artistic style.

Key concept: Domain translation — changing the visual domain of an image rather than simply applying a conventional filter.

2. VGG19 Neural Style Transfer

Neural Style Transfer uses a pretrained VGG19 network to extract content and style features from images.

Content image: Original landscape photograph.

Style image: Vincent van Gogh's The Starry Night.

Feature extraction: VGG19 extracts representations from multiple network layers.

Content loss: Encourages the output to retain the original image's structure.

Style loss: Uses Gram matrices to represent and match artistic texture and feature correlations.

Optimization: Iteratively adjusts the target image to combine the content and style representations.

Methodology

Load the required Python libraries and pretrained models.

Download and prepare a landscape photograph.

Apply the pretrained photo-to-Monet CycleGAN generator.

Load VGG19 as a feature extractor.

Load the artistic style reference image.

Extract content and style features.

Optimize the target image using content loss and style loss.

Display the outputs and compare both methods.

Results and Observations

The experiment demonstrates two different approaches to artistic image transformation.

CycleGAN: Translates the photograph into a Monet-inspired visual domain.

VGG19 Neural Style Transfer: Attempts to retain the original landscape's structure while incorporating the textures and visual patterns of the selected artwork.

The two methods differ in their objectives, feature representations, and image-generation processes.

Applications

Digital art and creative design

Photo stylization

Generative AI experimentation

Computer vision research

Artistic image transformation

Content creation

How to Run

This notebook is designed for a Python environment with internet access. Google Colab is a convenient option because the notebook downloads external resources and uses a GitHub-hosted CycleGAN implementation.

Open GEN_AI_ASSIGNMENT_6.ipynb in Google Colab or Jupyter Notebook.

Install the dependencies specified in the notebook.

Run the setup cells to download the required code, images, and pretrained model weights.

Execute the CycleGAN section.

Execute the VGG19 Neural Style Transfer section.

Review the final side-by-side image comparison.

Dependencies

The notebook uses the following libraries:

pip install torch torchvision pillow requests matplotlib numpy

It also clones the official CycleGAN implementation and installs its additional requirements as specified in the notebook.

Note: The pretrained CycleGAN checkpoint must be available at the path expected by the notebook. External image URLs and model files must also remain accessible.

Limitations

The output quality depends on the pretrained models and their learned representations.

CycleGAN may alter image details while translating between visual domains.

Neural Style Transfer requires iterative optimization and can be computationally expensive.

Results depend on the selected content image, style reference, and optimization settings.

The two methods serve different purposes, so their outputs are not directly equivalent.

Conclusion

This project demonstrates how generative AI can transform photographs into artistic images using two different deep-learning methods. CycleGAN performs artistic domain translation, whereas VGG19 Neural Style Transfer combines content features from a photograph with style features from an artwork. Comparing these methods provides practical insight into image-to-image translation and neural style transfer.

