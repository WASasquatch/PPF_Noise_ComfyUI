import torch
import torch.nn.functional as F
import math

import nodes

def normalize(latent, target_min=0.0, target_max=1.0):
    min_val = latent.min()
    max_val = latent.max()
    normalized = (latent - min_val) / (max_val - min_val)
    scaled = normalized * (target_max - target_min) + target_min
    return scaled
    
def perlin_power_fractal_batch(batch_size, width, height, X, Y, Z, frame, device='cpu', evolution_factor=0.1, octaves=4, persistence=0.5, lacunarity=2.0, exponent=4.0, scale=100, brightness=0.0, contrast=0.0, seed=None, min_clamp=0.0, max_clamp=1.0):
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
        torch.Tensor: A tensor containing the generated images in the shape (batch_size, height, width, 1).
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

    noise_map = torch.zeros(batch_size, height, width, dtype=torch.float32, device=device)

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

        noise_map += noise_values.squeeze(-1) * amplitude

    
    latent = normalize(noise_map, min_clamp, max_clamp)
    latent = (latent + brightness) * (1.0 + contrast)
    latent = latent.unsqueeze(-1)
    
    return latent
    
blending_modes = {
    'add': lambda a, b, factor: (a * factor + b * factor),
    'multiply': lambda a, b, factor: (a * factor * b * factor),
    'divide': lambda a, b, factor: (a * factor / b * factor),
    'subtract': lambda a, b, factor: (a * factor - b * factor),
    'overlay': lambda a, b, factor: (2 * a * b + a**2 - 2 * a * b * a) * factor if torch.all(b < 0.5) else (1 - 2 * (1 - a) * (1 - b)) * factor,
    'screen': lambda a, b, factor: (1 - (1 - a) * (1 - b) * (1 - factor)),
    'difference': lambda a, b, factor: (abs(a - b) * factor),
    'exclusion': lambda a, b, factor: ((a + b - 2 * a * b) * factor),
    'hard_light': lambda a, b, factor: (2 * a * b * (a < 0.5).float() + (1 - 2 * (1 - a) * (1 - b)) * (a >= 0.5).float()) * factor,
    'linear_dodge': lambda a, b, factor: (torch.clamp(a + b, 0, 1) * factor),
    'soft_light': lambda a, b, factor: (2 * a * b + a ** 2 - 2 * a * b * a if b < 0.5 else 2 * a * (1 - b) + torch.sqrt(a) * (2 * b - 1)) * factor,
    'random': lambda a, b, factor: (torch.rand_like(a) * a * factor + torch.rand_like(b) * b * factor)
}
    
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
                "resampling": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],),
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
                "clamp_min": ("FLOAT", {"default": 0.0, "max": 10.0, "min": -10.0, "step": 0.01}),
                "clamp_max": ("FLOAT", {"default": 1.0, "max": 10.0, "min": -10.0, "step": 0.01}),
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
    
    def power_fractal_latent(self, batch_size, width, height, resampling, X, Y, Z, evolution, frame, scale, octaves, persistence, lacunarity, exponent, brightness, contrast, clamp_min, clamp_max, seed, device, optional_vae=None):
                    
        color_intensity = 1
        masking_intensity = 1

        channel_tensors = []
        for i in range(batch_size):
            nseed = seed + i * 12
            rgb_noise_maps = []
            
            rgb_image = torch.zeros(4, height, width)
            
            for j in range(3):
                rgba_noise_map = self.generate_noise_map(width, height, X, Y, Z, frame, device, evolution, octaves, persistence, lacunarity, exponent, scale, brightness, contrast, nseed + j, clamp_min, clamp_max)
                rgb_noise_map = rgba_noise_map.squeeze(-1)
                rgb_noise_map *= color_intensity
                rgb_noise_map *= masking_intensity
                
                rgb_image[j] = rgb_noise_map
                
            rgb_image[3] = torch.ones(height, width)
            channel_tensors.append(rgb_image)
            
        tensors = torch.stack(channel_tensors)
        tensors = normalize(tensors)
        
        if optional_vae is None:
            latents = F.interpolate(tensors, size=((height // 8), (width // 8)), mode=resampling)
            return {'samples': latents}, tensors.permute(0, 2, 3, 1)
            
        encoder = nodes.VAEEncode()
        
        latents = []
        for tensor in tensors:
            tensor = tensor.unsqueeze(0)
            tensor = tensor.permute(0, 2, 3, 1)
            latents.append(encoder.encode(optional_vae, tensor)[0]['samples'])
            
        latents = torch.cat(latents)
        
        return {'samples': latents}, tensors.permute(0, 2, 3, 1)
        
    def generate_noise_map(self, width, height, X, Y, Z, frame, device, evolution, octaves, persistence, lacunarity, exponent, scale, brightness, contrast, seed, clamp_min, clamp_max):
        noise_map = perlin_power_fractal_batch(1, width, height, X, Y, Z, frame, device, evolution, octaves, persistence, lacunarity, exponent, scale, brightness, contrast, seed, clamp_min, clamp_max)
        return noise_map
        
class WAS_PFN_Blend_Latents:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a": ("LATENT",),
                "latent_b": ("LATENT",),
                "operation": (sorted(list(blending_modes.keys())),),
                "blend_ratio": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "mask": ("MASK",),
                "set_noise_mask": (["false", "true"],),
                "normalize": (["false", "true"],),
                "clamp_min": ("FLOAT", {"default": 0.0, "max": 10.0, "min": -10.0, "step": 0.01}),
                "clamp_max": ("FLOAT", {"default": 1.0, "max": 10.0, "min": -10.0, "step": 0.01}),
                "latent2rgb_preview": (["false", "true"],),
            }
        }

    RETURN_TYPES = ("LATENT","IMAGE",)
    RETURN_NAMES = ("latents", "previews")
    FUNCTION = "latent_blend"
    CATEGORY = "latent"

    def latent_blend(self, latent_a, latent_b, operation, blend_ratio, blend_strength, mask=None, set_noise_mask=None, normalize=None, clamp_min=None, clamp_max=None, latent2rgb_preview=None):
        latent_a_rgb = latent_a["samples"][:, :-1]
        latent_b_rgb = latent_b["samples"][:, :-1]

        alpha_a = latent_a["samples"][:, -1:]
        alpha_b = latent_b["samples"][:, -1:]
        
        blended_rgb = self.blend_latents(latent_a_rgb, latent_b_rgb, operation, blend_ratio, blend_strength, clamp_min, clamp_max)
        blended_alpha = torch.ones_like(blended_rgb[:, :1])
        blended_latent = torch.cat((blended_rgb, blended_alpha), dim=1)
        
        if latent2rgb_preview and latent2rgb_preview == "true":
            l2rgb = torch.tensor([
                #   R     G      B
                [0.298, 0.207, 0.208],  # L1
                [0.187, 0.286, 0.173],  # L2
                [-0.158, 0.189, 0.264],  # L3
                [-0.184, -0.271, -0.473],  # L4
            ], device=blended_latent.device)
            tensors = torch.einsum('...lhw,lr->...rhw', blended_latent.float(), l2rgb)
            tensors = ((tensors + 1) / 2).clamp(0, 1)
            tensors = tensors.movedim(1, -1)          
        else:
            tensors = blended_latent.permute(0, 2, 3, 1)

        if mask is not None:
            blend_mask = self.transform_mask(mask, latent_a["samples"].shape)
            blended_latent = blend_mask * blended_latent + (1 - blend_mask) * latent_a["samples"]
            if set_noise_mask == 'true':
                return ({"samples": blended_latent, "noise_mask": mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))}, tensors)
            else:
                return ({"samples": blended_latent}, tensors)
        else:
            return ({"samples": blended_latent}, tensors)
            
    def blend_latents(self, latent1, latent2, mode='add', blend_percentage=0.5, blend_strength=0.5, mask=None, clamp_min=0.0, clamp_max=1.0):
        blend_func = blending_modes.get(mode)
        if blend_func is None:
            raise ValueError("Unsupported blending mode. Please choose from the supported modes.")
        
        blend_factor1 = blend_percentage
        blend_factor2 = 1 - blend_percentage

        blended_latent = blend_func(latent1, latent2, blend_strength * blend_factor1)

        if normalize and normalize == "true":
            blended_latent = normalize(blended_latent, clamp_min, clamp_max)
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
    "Perlin Power Fractal Latent (PPF Noise)": WAS_PFN_Latent,
    "Blend Latents (PPF Noise)": WAS_PFN_Blend_Latents
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Perlin Power Fractal Latent (PPF Noise)": "Perlin Power Fractal Noise (PPF Noise)",
    "Blend Latents (PPF Noise)": "Blend Latents (PPF Noise)"
}

