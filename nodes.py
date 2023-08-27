import numpy as np
from PIL import Image
import torch
from numba import jit
import random

import nodes

def perlin_power_fractal_batch(batch_size, width, height, octaves, persistence, lacunarity, exponent, scale, seed=None):
    """
    Generate a batch of images with a Perlin power fractal effect.

    Parameters:
        batch_size (int): Number of noisy tensors to generate in the batch.
        width (int): Width of each tensor in pixels.
        height (int): Height of each image in pixels.
        octaves (int): Number of octaves for fractal generation. Controls the level of detail and complexity in the output.
            Lower values (0-3) create smoother patterns, while higher values (6-8) create more intricate and rough patterns.
        persistence (float): Persistence parameter for fractal generation. Determines the amplitude decrease of each octave.
            Higher values (0.5-0.9) result in more defined and contrasted patterns, while lower values create smoother patterns.
        lacunarity (float): Lacunarity parameter for fractal generation. Controls the increase in frequency from one octave to the next.
            Higher values (2.0-3.0) create more detailed and finer patterns, while lower values create coarser patterns.
        exponent (int): Exponent applied to the noise values. Adjusting this parameter controls the overall intensity and contrast of the output.
            Higher values (>1) emphasize the differences between noisy elements, resulting in more distinct features.
        scale (int): Scaling factor for frequency of noise. Larger values produce smaller, more detailed patterns, while smaller values create larger patterns.
        seed (int, optional): Seed for random number generation. If None, uses random seeds for each batch.

    Returns:
        torch.Tensor: A tensor containing the generated images in the shape (batch_size, 4, height, width).
    """
    
    @jit(nopython=True)
    def fade(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    @jit(nopython=True)
    def lerp(t, a, b):
        return a + t * (b - a)

    @jit(nopython=True)
    def grad(hash, x, y, z):
        h = hash & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h == 12 or h == 14 else z)
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    @jit(nopython=True)
    def noise(x, y, z, p):
        X = np.int32(np.floor(x)) & 255
        Y = np.int32(np.floor(y)) & 255
        Z = np.int32(np.floor(z)) & 255

        x -= np.floor(x)
        y -= np.floor(y)
        z -= np.floor(z)

        u = fade(x)
        v = fade(y)
        w = fade(z)

        A = p[X] + Y
        AA = p[A] + Z
        AB = p[A + 1] + Z
        B = p[X + 1] + Y
        BA = p[B] + Z
        BB = p[B + 1] + Z

        return lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)),
                                lerp(u, grad(p[AB], x, y - 1, z), grad(p[BB], x - 1, y - 1, z))),
                    lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1), grad(p[BA + 1], x - 1, y, z - 1)),
                                lerp(u, grad(p[AB + 1], x, y - 1, z - 1), grad(p[BB + 1], x - 1, y - 1, z - 1))))

    noise_maps = []

    for i in range(batch_size):
        unique_seed = seed + i if seed is not None else None

        np.random.seed(unique_seed)
        p = np.arange(256, dtype=np.int32)
        np.random.shuffle(p)
        p = np.concatenate((p, p))

        noise_map_r = np.zeros((height, width))
        noise_map_g = np.zeros((height, width))
        noise_map_b = np.zeros((height, width))
        noise_map_a = np.zeros((height, width))

        amplitude = 1.0
        total_amplitude = 0.0

        for octave in range(octaves):
            frequency = lacunarity ** octave
            amplitude *= persistence
            total_amplitude += amplitude

            for y in range(height):
                for x in range(width):
                    nx = x / scale * frequency
                    ny = y / scale * frequency
                    noise_value_r = noise(nx, ny, 0, p) * amplitude ** exponent
                    noise_value_g = noise(nx + 1000, ny + 1000, 0, p) * amplitude ** exponent
                    noise_value_b = noise(nx + 2000, ny + 2000, 0, p) * amplitude ** exponent
                    noise_value_a = noise(nx + 3000, ny + 3000, 0, p) * amplitude ** exponent

                    current_value_r = noise_map_r[y, x]
                    current_value_g = noise_map_g[y, x]
                    current_value_b = noise_map_b[y, x]
                    current_value_a = noise_map_a[y, x]

                    noise_map_r[y, x] = current_value_r + noise_value_r
                    noise_map_g[y, x] = current_value_g + noise_value_g
                    noise_map_b[y, x] = current_value_b + noise_value_b
                    noise_map_a[y, x] = current_value_a + noise_value_a

        min_value_r = np.min(noise_map_r)
        max_value_r = np.max(noise_map_r)
        min_value_g = np.min(noise_map_g)
        max_value_g = np.max(noise_map_g)
        min_value_b = np.min(noise_map_b)
        max_value_b = np.max(noise_map_b)
        min_value_a = np.min(noise_map_a)
        max_value_a = np.max(noise_map_a)

        noise_map_r = np.interp(noise_map_r, (min_value_r, max_value_r), (0, 255)).astype(np.uint8)
        noise_map_g = np.interp(noise_map_g, (min_value_g, max_value_g), (0, 255)).astype(np.uint8)
        noise_map_b = np.interp(noise_map_b, (min_value_b, max_value_b), (0, 255)).astype(np.uint8)
        noise_map_a = np.interp(noise_map_a, (min_value_a, max_value_a), (0, 255)).astype(np.uint8)

        noise_map = np.stack((noise_map_r, noise_map_g, noise_map_b, noise_map_a), axis=-1)
        noise_maps.append(noise_map)

    noise_maps_array = np.array(noise_maps, dtype=np.uint8)
    image_tensor_batch = torch.tensor(noise_maps_array, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    
    return image_tensor_batch.view(batch_size, 4, height, width)
    
    
# COMFYUI NODES

class WAS_PFN_Latent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"default": 1, "max": 64, "min": 1, "step": 1}),
                "width": ("INT", {"default": 512, "max": 8192, "min": 64, "step": 1}),
                "height": ("INT", {"default": 512, "max": 8192, "min": 64, "step": 1}),
                "scale": ("INT", {"default": 100, "max": 2048, "min": 2, "step": 1}),
                "octaves": ("INT", {"default": 8, "max": 8, "min": 0, "step": 1}),
                "persistence": ("FLOAT", {"default": 1.0, "max": 100.0, "min": 0.01, "step": 0.01}),
                "lacunarity": ("FLOAT", {"default": 2.0, "max": 100.0, "min": 0.01, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 4.0, "max": 100.0, "min": 0.01, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), 
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "power_fractal_latent"

    CATEGORY = "latent/noise"

    def power_fractal_latent(self, batch_size, width, height, scale, octaves, persistence, lacunarity, exponent, seed):
            
        width = width // 8
        height = height // 8
        
        noise_maps = perlin_power_fractal_batch(batch_size, width, height, octaves, persistence, lacunarity, exponent, scale, seed)       
        
        return ({'samples': noise_maps}, )
        

NODE_CLASS_MAPPINGS = {
    "WAS_PFN_Latent": WAS_PFN_Latent
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WAS_PFN_Latent": "Perlin Power Fractal Noise"
}

