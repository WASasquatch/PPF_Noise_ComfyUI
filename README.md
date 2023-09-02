# Perlin Power Fractal Noise for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
 Perlin Power Fractal Noisey Latents

<img src="https://github.com/WASasquatch/PPF_Noise_ComfyUI/assets/1151589/56d3c514-0462-4b1c-adaa-7fe6977f1bcd" width="600">

# Power Fractal Latent Generator

Generate a batch of images with a Perlin power fractal effect.

---

# Installation
 - Clone the repo to `ComfyUI/custom_nodes`. Torch versions do not need requirements.txt installed.
   - If you are using previous non-torch builds, run the requirements.txt against your ComfyUI Python Environment
     - ***ComfyUI Standalone Portable example:*** `C:\ComfyUI_windows_portable\python_embeded\python.exe -s -m pip install -r "C:\ComfyUI_windows_portable\custom_nodes\PPF_Noise_ComfyUI\requirements.txt"`
    
---

## **Perlin Power Fractal Noise** Parameters

### Required:
- `batch_size` (int): Number of noisy tensors to generate in the batch.
    - Range: [1, 64]
- `width` (int): Width of each tensor in pixels.
    - Range: [64, 8192]
- `height` (int): Height of each image in pixels.
- `resampling` (string): This parameter determines the resampling method used for scaling noise to the latent size. Choose from the following options:
    - "**nearest-exact**": Nearest-Exact Resampling:
        - Nearest-neighbor resampling selects the pixel value from the nearest source pixel, resulting in a blocky, pixelated appearance. It preserves the exact values without interpolation.
    - "**bilinear**": Bilinear Resampling:
        - Bilinear interpolation takes a weighted average of the four nearest source pixels, producing smoother transitions between pixels. It's a good choice for general image resizing.
    - "**area**": Area Resampling (Antialiasing):
        - Resampling using pixel area relation, also known as antialiasing, computes pixel values based on the areas of contributing source pixels. It reduces aliasing artifacts and is suitable for preserving fine details.
    - "**bicubic**": Bicubic Resampling:
        - Bicubic interpolation uses a cubic polynomial to compute pixel values based on the 16 nearest source pixels. It provides smoother transitions and better detail preservation, suitable for high-quality resizing.
    - "**bislerp**": Bislerp Resampling (Bilinear Sinc Interpolation):
        - Bislerp interpolation combines bilinear simplicity with sinc function interpolation, resulting in high-quality resizing with reduced artifacts. It offers a balance between quality and computational cost.
- `X` (float): X-coordinate offset for noise sampling.
    - Range: [-99999999, 99999999]
- `Y` (float): Y-coordinate offset for noise sampling.
    - Range: [-99999999, 99999999]
- `Z` (float): Z-coordinate offset for noise sampling.
    - Range: [-99999999, 99999999]
- `frame` (int): The current frame number for time evolution.
    - Range: [0, 99999999]
- `evolution_factor` (float): Factor controlling time evolution. Determines how much the noise evolves over time based on the batch index.
    - Range: [0.0, 1.0]
- `octaves` (int): Number of octaves for fractal generation. Controls the level of detail and complexity in the output.
    - Range: [1, 8]
- `persistence` (float): Persistence parameter for fractal generation. Determines the amplitude decrease of each octave.
    - Range: [0.01, 23.0]
- `lacunarity` (float): Lacunarity parameter for fractal generation. Controls the increase in frequency from one octave to the next.
    - Range: [0.01, 99.0]
- `exponent` (float): Exponent applied to the noise values. Adjusting this parameter controls the overall intensity and contrast of the output.
    - Range: [0.01, 38.0]
- `scale` (float): Scaling factor for frequency of noise. Larger values produce smaller, more detailed patterns, while smaller values create larger patterns.
    - Range: [2, 2048]
- `brightness` (float): Adjusts the overall brightness of the generated noise.
    - -1.0 makes the noise completely black.
    - 0.0 has no effect on brightness.
    - 1.0 makes the noise completely white.
    - Range: [-1.0, 1.0]
- `contrast` (float): Adjusts the contrast of the generated noise.
    - -1.0 reduces contrast, enhancing the difference between dark and light areas.
    - 0.0 has no effect on contrast.
    - 1.0 increases contrast, enhancing the difference between dark and light areas.
    - Range: [-1.0, 1.0]
- `clamp_min` (float): The floor range of the noise
  - Range: [-10.0, 10]
- `clamp_max` (float): The ceiling range of the noise
 - Range: [-10, 10]
- `seed` (int, optional): Seed for random number generation. If None, uses random seeds for each batch.
    - Range: [0, 0xffffffffffffffff]
- `device` (string): Specify the device to generate noise on, either "cpu" or "cuda".
### Optional:
- `optional_vae` (VAE, optional): The optional VAE for encoding the noise.
### Returns
- `tuple` (torch.Tensor [latent], torch.Tensor [image])

---

## **WAS_PFN_Blend_Latents** Parameters

This class provides a method for blending two latent tensors.

### Required:
- `latent_a` (LATENT, required): The first input latent tensor to be blended.
- `latent_b` (LATENT, required): The second input latent tensor to be blended.
- `operation` (string, required): The blending operation to apply. Choose from the following options:
    - "add": Additive blending.
        - Adds the values of `latent_a` and `latent_b`.
    - "multiply": Multiplicative blending.
        - Multiplies the values of `latent_a` and `latent_b`.
    - "divide": Division blending.
        - Divides the values of `latent_a` by `latent_b`.
    - "subtract": Subtraction blending.
        - Subtracts the values of `latent_b` from `latent_a`.
    - "overlay": Overlay blending.
        - Applies an overlay blending effect.
    - "hard_light": Hard light blending.
        - Applies a hard light blending effect.
    - "soft_light": Soft light blending.
        - Applies a soft light blending effect.
    - "screen": Screen blending.
        - Applies a screen blending effect.
    - "linear_dodge": Linear dodge blending.
        - Applies a linear dodge blending effect.
    - "difference": Difference blending.
        - Computes the absolute difference between `latent_a` and `latent_b`.
    - "exclusion": Exclusion blending.
        - Applies an exclusion blending effect.
    - "random": Random noise blending.
        - Applies a random noise blending effect.
- `blend_ratio` (FLOAT, required): The blend ratio between `latent_a` and `latent_b`. 
    - Default: 0.5
    - Range: [0.01, 1.0]
- `blend_strength` (FLOAT, required): The strength of the blending operation.
    - Default: 1.0
    - Range: [0.0, 100.0]
### Optional:
- `mask` (MASK, optional): An optional mask tensor to control the blending region.
- `set_noise_mask` (string, optional): Whether to set the noise mask. Choose from "false" or "true".
- `normalize` (string, optional): Whether to normalize the resulting latent tensor. Choose from "false" or "true".
- `clamp_min` (FLOAT, optional): The minimum clamping range for the output.
    - Default: 0.0
    - Range: [-10.0, 10.0]
- `clamp_max` (FLOAT, optional): The maximum clamping range for the output.
    - Default: 1.0
    - Range: [-10.0, 10.0]
### Returns
- `tuple` (LATENT,): A tuple containing the blended latent tensor.
