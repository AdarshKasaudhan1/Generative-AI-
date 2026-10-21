# Latent Space Interpolation and Image Generation Using StyleGAN2-ADA

## Overview

This project explores face image generation using StyleGAN2-ADA, a pretrained generative adversarial network (GAN). It investigates how latent vectors influence generated images through latent space interpolation and modifications to individual latent dimensions.

## Objectives

- Generate face images using a pretrained StyleGAN2-ADA model.
- Perform linear interpolation between two latent vectors.
- Observe visual changes in generated images during interpolation.
- Modify an individual latent dimension and examine its effect on the output.
- Understand how latent representations influence image generation.

## Technologies Used

- Python
- Jupyter Notebook
- StyleGAN2-ADA
- NumPy
- Matplotlib
- PyTorch

## Methodology

1. Load the pretrained StyleGAN2-ADA model.
2. Generate latent vectors as inputs to the generator.
3. Produce face images from the selected latent vectors.
4. Apply linear interpolation between two latent vectors.
5. Generate intermediate images to observe changes between the endpoints.
6. Modify one latent dimension across different values and compare the generated outputs.

## Key Observations

The experiment demonstrated how StyleGAN2-ADA generates face images from latent vectors. Interpolating between two latent vectors produced a sequence of images with gradual visual differences between the starting and ending faces. Modifying an individual latent dimension also produced variations in the generated images, illustrating how latent-space values can influence facial appearance.

## Results

The experiment illustrates two approaches to exploring a generative model's latent space:

- **Latent Space Interpolation:** Examines transitions between images generated from two latent vectors.
- **Latent Dimension Manipulation:** Examines how changing one latent value affects the generated image.

## Applications

- Synthetic face image generation
- Generative AI experimentation
- Latent representation analysis
- Computer vision research
- Exploration of image transformations using GANs

## How to Run

1. Clone or download this repository.
2. Open the `.ipynb` notebook in Jupyter Notebook or Google Colab.
3. Install the dependencies required by the notebook.
4. Ensure the pretrained StyleGAN2-ADA model and its required files are accessible.
5. Run the notebook cells in sequence to generate images and explore latent-space transformations.

## Limitations

- Results depend on the pretrained model and its learned representations.
- Changing a single latent dimension may affect multiple visual characteristics.
- The experiment does not establish that each latent dimension corresponds to one specific facial feature.
- Generated results may vary depending on the model configuration and latent vectors.

## Conclusion

This project demonstrates how latent vectors can be used to generate and manipulate face images with StyleGAN2-ADA. Through interpolation and individual dimension changes, it provides a practical introduction to latent-space exploration in generative AI.

## Author

Adarsh Kasaudhan
