import torch
import math

import nodes

def perlin_power_fractal_batch(batch_size, width, height, X, Y, Z, frame, device='cpu', evolution_factor=0.1, octaves=4, persistence=0.5, lacunarity=2.0, exponent=4.0, scale=100, brightness=0.0, contrast=0.0, amplify_latent=False, seed=None):
    """
    Generate a batch of images with a Perlin power fractal effect.

    Parameters:
        batch_size (int): Number of noisy tensors to generate in the batch.
            Range: [1, 64]
        width (int): Width of each tensor in pixels.
            Range: [64, 8192]
        height (int): Height of each image in pixels.
            Range: [64, 8192]
        X (float): X-coordinate offset for noise sampling.
            Range: [-99999999, 99999999]
        Y (float): Y-coordinate offset for noise sampling.
            Range: [-99999999, 99999999]
        Z (float): Z-coordinate offset for noise sampling.
            Range: [-99999999, 99999999]
        frame (int): The current frame number for time evolution.
            Range: [0, 99999999]
        evolution_factor (float): Factor controlling time evolution. Determines how much the noise evolves over time based on the batch index.
            Range: [0.0, 1.0]
        octaves (int): Number of octaves for fractal generation. Controls the level of detail and complexity in the output.
            Range: [1, 8]
        persistence (float): Persistence parameter for fractal generation. Determines the amplitude decrease of each octave.
            Range: [0.01, 23.0]
        lacunarity (float): Lacunarity parameter for fractal generation. Controls the increase in frequency from one octave to the next.
            Range: [0.01, 99.0]
        exponent (float): Exponent applied to the noise values. Adjusting this parameter controls the overall intensity and contrast of the output.
            Range: [0.01, 38.0]
        scale (float): Scaling factor for frequency of noise. Larger values produce smaller, more detailed patterns, while smaller values create larger patterns.
            Range: [2, 2048]
        brightness (float): Adjusts the overall brightness of the generated noise.
            - -1.0 makes the noise completely black.
            - 0.0 has no effect on brightness.
            - 1.0 makes the noise completely white.
            Range: [-1.0, 1.0]
        contrast (float): Adjusts the contrast of the generated noise.
            - -1.0 reduces contrast, enhancing the difference between dark and light areas.
            - 0.0 has no effect on contrast.
            - 1.0 increases contrast, enhancing the difference between dark and light areas.
            Range: [-1.0, 1.0]
        seed (int, optional): Seed for random number generation. If None, uses random seeds for each batch.
            Range: [0, 0xffffffffffffffff]

    Returns:
        torch.Tensor: A tensor containing the generated images in the shape (batch_size, 4, height, width).
    """
    def fade(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def lerp(t, a, b):
        return a + t * (b - a)

    def grad(hash, x, y, z):
        h = hash & 15
        u = torch.where(h < 8, x, y)
        v = torch.where(h < 4, y, torch.where((h == 12) | (h == 14), x, z))
        return torch.where(h & 1 == 0, u, -u) + torch.where(h & 2 == 0, v, -v)

    def noise(x, y, z, p):
        X = (x.floor() % 255).to(torch.int32)
        Y = (y.floor() % 255).to(torch.int32)
        Z = (z.floor() % 255).to(torch.int32)

        x -= x.floor()
        y -= y.floor()
        z -= z.floor()

        u = fade(x)
        v = fade(y)
        w = fade(z)

        A = p[X] + Y
        AA = p[A] + Z
        AB = p[A + 1] + Z
        B = p[X + 1] + Y
        BA = p[B] + Z
        BB = p[B + 1] + Z

        r = lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)),
                          lerp(u, grad(p[AB], x, y - 1, z), grad(p[BB], x - 1, y - 1, z))),
                 lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1), grad(p[BA + 1], x - 1, y, z - 1)),
                          lerp(u, grad(p[AB + 1], x, y - 1, z - 1), grad(p[BB + 1], x - 1, y - 1, z - 1))))

        return r

    device = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'

    unique_seed = seed if seed is not None else torch.randint(0, 10000, (1,)).item()
    torch.manual_seed(unique_seed)

    p = torch.randperm(max(width, height)**2, dtype=torch.int32, device=device)
    p = torch.cat((p, p))

    noise_map = torch.zeros(batch_size, height, width, 4, dtype=torch.float32, device=device)

    X = torch.arange(width, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) + X
    Y = torch.arange(height, dtype=torch.float32, device=device).unsqueeze(1).unsqueeze(0) + Y
    Z = evolution_factor * torch.arange(batch_size, dtype=torch.float32, device=device).unsqueeze(1).unsqueeze(1) + Z + frame

    for octave in range(octaves):
        frequency = lacunarity ** octave
        amplitude = persistence ** octave

        nx = (X + frame * evolution_factor) / scale * frequency
        ny = (Y + frame * evolution_factor) / scale * frequency
        nz = (Z + octave) / scale

        noise_values = noise(nx, ny, nz, p) * (amplitude ** exponent)

        noise_map += noise_values.unsqueeze(-1)

    latent = (noise_map + brightness) * (1.0 + contrast)
    latent = torch.clamp(latent, 0.0, 1.0)

    latent[..., -1] = 1.0

    if amplify_latent:
        amplification = 2.0
    else:
        amplification = 1.0
        
    print(f"Amplification: {amplification}")

    min_value = latent.min()
    max_value = latent.max()

    latent = (latent - min_value) / (max_value - min_value)

    latent = torch.zeros([batch_size, 4, height, width]) + (latent.permute(0, 3, 1, 2).to(device="cpu") * amplification)

    
    return latent
    
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
                "X": ("FLOAT", {"default": 0, "max": 99999999, "min": -99999999, "step": 0.01}),
                "Y": ("FLOAT", {"default": 0, "max": 99999999, "min": -99999999, "step": 0.01}),
                "Z": ("FLOAT", {"default": 0, "max": 99999999, "min": -99999999, "step": 0.01}),
                "evolution": ("FLOAT", {"default": 0.0, "max": 1.0, "min": 0.0, "step": 0.01}),
                "frame": ("INT", {"default": 0, "max": 99999999, "min": 0, "step": 1}),
                "scale": ("FLOAT", {"default": 5, "max": 2048, "min": 2, "step": 0.01}),
                "octaves": ("INT", {"default": 8, "max": 8, "min": 1, "step": 1}),
                "persistence": ("FLOAT", {"default": 1.0, "max": 23.0, "min": 0.01, "step": 0.01}),
                "lacunarity": ("FLOAT", {"default": 2.0, "max": 99.0, "min": 0.01, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 4.0, "max": 38.0, "min": 0.01, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "max": 1.0, "min": -1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 0.0, "max": 1.0, "min": -1.0, "step": 0.01}),
                "amplify_latent": (['false', 'true',],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), 
                "device": (["cpu", "cuda"],),
            },
            "optional": {
                "optional_vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT","IMAGE")
    RETURN_NAMES = ("latents","previews")
    FUNCTION = "power_fractal_latent"

    CATEGORY = "latent/noise"

    def power_fractal_latent(self, batch_size, width, height, X, Y, Z, evolution, frame, scale, octaves, persistence, lacunarity, exponent, brightness, contrast, amplify_latent, seed, device, optional_vae=None):
    
        if optional_vae == None:
            width = width // 8
            height = height // 8
            
        color_intensity = 2
        masking_intensity = 12
        amplify_latent = (amplify_latent == "true")

        modified_rgba_tensors = []
        for i in range(batch_size):
            nseed = seed + i * 12
            rgba_noise_maps = []
            
            for j in range(6):
                rgba_noise_map = self.generate_noise_map(width, height, X, Y, Z, frame, device, evolution, octaves, persistence, lacunarity, exponent, scale, brightness, contrast, amplify_latent, nseed + j)
                rgba_noise_maps.append(rgba_noise_map.squeeze(0))

            red_mask = torch.mean(rgba_noise_maps[3], dim=0, keepdim=True) * torch.mean(rgba_noise_maps[4], dim=0, keepdim=True) * color_intensity
            green_mask = torch.mean(rgba_noise_maps[3], dim=0, keepdim=True) * torch.mean(rgba_noise_maps[4], dim=0, keepdim=True) * color_intensity
            blue_mask = torch.mean(rgba_noise_maps[3], dim=0, keepdim=True) * torch.mean(rgba_noise_maps[4], dim=0, keepdim=True) * color_intensity
            
            min_noise_value = torch.min(rgba_noise_maps[5])
            max_noise_value = torch.max(rgba_noise_maps[5])

            desired_min_value = 0.4
            desired_max_value = 0.8

            scale_factor = (desired_max_value - desired_min_value) / (max_noise_value - min_noise_value)
            shift_factor = desired_min_value - min_noise_value * scale_factor

            masking_map = (rgba_noise_maps[5][0:1, :, :] * scale_factor + shift_factor)

            masking_map = torch.clamp(masking_map * masking_intensity, 0.0, 1.0)

            red_mask = red_mask * masking_map
            green_mask = green_mask * masking_map
            blue_mask = blue_mask * masking_map
            
            alpha_channel = torch.ones_like(rgba_noise_maps[0][0:1, :, :])

            modified_rgba_map = torch.stack([
                rgba_noise_maps[0][0:1, :, :] * red_mask,
                rgba_noise_maps[1][1:2, :, :] * green_mask,
                rgba_noise_maps[2][2:3, :, :] * blue_mask,
                alpha_channel
            ], dim=3)

            modified_rgba_tensors.append(modified_rgba_map)
            
        tensors = torch.cat(modified_rgba_tensors, dim=0)
        
        if optional_vae == None:
            latents = tensors.permute(0, 3, 1, 2)
            return ({'samples': latents}, tensors)
            
        encoder = nodes.VAEEncode()
        
        latents = []
        for tensor in tensors:
            latents.append(encoder.encode(optional_vae, tensor.unsqueeze(0))[0]['samples'])
            
        latents = torch.cat(latents)

        return ({'samples': latents}, tensors)
        
    def generate_noise_map(self, width, height, X, Y, Z, frame, device, evolution, octaves, persistence, lacunarity, exponent, scale, brightness, contrast, amplify_latent, seed):
        return perlin_power_fractal_batch(1, width, height, X, Y, Z, frame, device, evolution, octaves, persistence, lacunarity, exponent, scale, brightness, contrast, amplify_latent, seed)
        
class WAS_PFN_Blend_Latents:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a": ("LATENT",),
                "latent_b": ("LATENT",),
                "operation": (["add", "multiply", "divide", "subtract", "overlay", "hard_light", "soft_light", "screen", "linear_dodge", "difference", "exclusion", "random"],),
                "blend_ratio": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "mask": ("MASK",),
                "set_noise_mask": (["false", "true"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "latent_blend"
    CATEGORY = "latent"

    def latent_blend(self, latent_a, latent_b, operation, blend_ratio, blend_strength, mask=None, set_noise_mask=None):
        blended_latent = self.blend_latents(latent_a["samples"], latent_b["samples"], operation, blend_ratio, blend_strength)
        if mask is not None:
            blend_mask = self.transform_mask(mask, latent_a["samples"].shape)
            blended_latent = blend_mask * blended_latent + (1 - blend_mask) * latent_a["samples"]
            if set_noise_mask == 'true':
                return ({"samples": blended_latent, "noise_mask": mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))}, )
            else:
                return ({"samples": blended_latent}, )
        else:
            return ({"samples": blended_latent}, )

    def blend_latents(self, latent1, latent2, mode='add', blend_percentage=0.5, blend_strength=0.5, mask=None):
        def overlay_blend(latent1, latent2, blend_factor):
            low = 2 * latent1 * latent2
            high = 1 - 2 * (1 - latent1) * (1 - latent2)
            blended_latent = (latent1 * blend_factor) * low + (latent2 * blend_factor) * high
            return blended_latent

        def screen_blend(latent1, latent2, blend_factor):
            inverted_latent1 = 1 - latent1
            inverted_latent2 = 1 - latent2
            blended_latent = 1 - (inverted_latent1 * inverted_latent2 * (1 - blend_factor))
            return blended_latent

        def difference_blend(latent1, latent2, blend_factor):
            blended_latent = abs(latent1 - latent2) * blend_factor
            return blended_latent

        def exclusion_blend(latent1, latent2, blend_factor):
            blended_latent = (latent1 + latent2 - 2 * latent1 * latent2) * blend_factor
            return blended_latent

        def hard_light_blend(latent1, latent2, blend_factor):
            blended_latent = torch.where(latent2 < 0.5, 2 * latent1 * latent2, 1 - 2 * (1 - latent1) * (1 - latent2)) * blend_factor
            return blended_latent

        def linear_dodge_blend(latent1, latent2, blend_factor):
            blended_latent = torch.clamp(latent1 + latent2, 0, 1) * blend_factor
            return blended_latent

        def soft_light_blend(latent1, latent2, blend_factor):
            low = 2 * latent1 * latent2 + latent1 ** 2 - 2 * latent1 * latent2 * latent1
            high = 2 * latent1 * (1 - latent2) + torch.sqrt(latent1) * (2 * latent2 - 1)
            blended_latent = (latent1 * blend_factor) * low + (latent2 * blend_factor) * high
            return blended_latent

        def random_noise(latent1, latent2, blend_factor):
            noise1 = torch.randn_like(latent1)
            noise2 = torch.randn_like(latent2)
            noise1 = (noise1 - noise1.min()) / (noise1.max() - noise1.min())
            noise2 = (noise2 - noise2.min()) / (noise2.max() - noise2.min())
            blended_noise = (latent1 * blend_factor) * noise1 + (latent2 * blend_factor) * noise2
            blended_noise = torch.clamp(blended_noise, 0, 1)
            return blended_noise
            
        def normalize(latent):
            return (latent - latent.min()) / (latent.max() - latent.min())
                
        blend_factor1 = blend_percentage
        blend_factor2 = 1 - blend_percentage

        if mode == 'add':
            blended_latent = (latent1 * blend_strength * blend_factor1) + (latent2 * blend_strength * blend_factor2)
        elif mode == 'multiply':
            blended_latent = (latent1 * blend_strength * blend_factor1) * (latent2 * blend_strength * blend_factor2)
        elif mode == 'divide':
            blended_latent = (latent1 * blend_strength * blend_factor1) / (latent2 * blend_strength * blend_factor2)
        elif mode == 'subtract':
            blended_latent = (latent1 * blend_strength * blend_factor1) - (latent2 * blend_strength * blend_factor2)
        elif mode == 'overlay':
            blended_latent = overlay_blend(latent1, latent2, blend_strength * blend_factor1)
        elif mode == 'screen':
            blended_latent = screen_blend(latent1, latent2, blend_strength * blend_factor1)
        elif mode == 'difference':
            blended_latent = difference_blend(latent1, latent2, blend_strength * blend_factor1)
        elif mode == 'exclusion':
            blended_latent = exclusion_blend(latent1, latent2, blend_strength * blend_factor1)
        elif mode == 'hard_light':
            blended_latent = hard_light_blend(latent1, latent2, blend_strength * blend_factor1)
        elif mode == 'linear_dodge':
            blended_latent = linear_dodge_blend(latent1, latent2, blend_strength * blend_factor1)
        elif mode == 'soft_light':
            blended_latent = soft_light_blend(latent1, latent2, blend_strength * blend_factor1)
        elif mode == 'random':
            blended_latent = random_noise(latent1, latent2, blend_strength * blend_factor1)
        else:
            raise ValueError("Unsupported blending mode. Please choose from 'add', 'multiply', 'divide', 'subtract', 'overlay', 'screen', 'difference', 'exclusion', 'hard_light', 'linear_dodge', 'soft_light', 'custom_noise'.")

        blended_latent = normalize(blended_latent)
        return blended_latent

    def transform_mask(self, mask, shape):
        mask = mask.view(-1, 1, mask.shape[-2], mask.shape[-1])
        resized_mask = torch.nn.functional.interpolate(mask, size=(shape[2], shape[3]), mode="bilinear")
        expanded_mask = resized_mask.expand(-1, shape[1], -1, -1)
        if expanded_mask.shape[0] < shape[0]:
            expanded_mask = expanded_mask.repeat((shape[0] - 1) // expanded_mask.shape[0] + 1, 1, 1, 1)[:shape[0]]
        del mask, resized_mask
        return expanded_mask
        
NODE_CLASS_MAPPINGS = {
    "WAS_PFN_Latent": WAS_PFN_Latent,
    "WAS_PFN_Blend_Latents": WAS_PFN_Blend_Latents
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WAS_PFN_Latent": "Perlin Power Fractal Noise",
    "WAS_PFN_Blend_Latents": "Blend Latents (PPF Noise)"
}

