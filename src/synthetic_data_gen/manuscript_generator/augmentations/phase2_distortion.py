# /manuscript_generator/augmentations/phase2_distortion.py

import numpy as np

from manuscript_generator.core.registry import register_augmentation
from manuscript_generator.utils.distribution_sampler import sample_from_distribution
from manuscript_generator.utils.geometry import apply_transform
from manuscript_generator.configs.base_config import Config
from manuscript_generator.core.textbox import TextBox

@register_augmentation("shear")
def apply_shear(points: np.ndarray, textbox: TextBox, config: Config, rng: np.random.Generator) -> np.ndarray:
    """Applies a shear transformation to the textbox points."""
    aug_config = config.textbox_distortion.shear
    sx = sample_from_distribution(aug_config.shear_factor_x, rng)
    sy = sample_from_distribution(aug_config.shear_factor_y, rng)
    shear_matrix = np.array([[1, sx], [sy, 1]])
    return apply_transform(points, shear_matrix)

@register_augmentation("stretch")
def apply_stretch(points: np.ndarray, textbox: TextBox, config: Config, rng: np.random.Generator) -> np.ndarray:
    """Applies non-uniform scaling to the textbox points."""
    aug_config = config.textbox_distortion.stretch
    stretch_x = sample_from_distribution(aug_config.stretch_factor_x, rng)
    stretch_y = sample_from_distribution(aug_config.stretch_factor_y, rng)
    stretch_matrix = np.array([[stretch_x, 0], [0, stretch_y]])
    return apply_transform(points, stretch_matrix)

@register_augmentation("warp_curl")
def apply_warp_curl(points: np.ndarray, textbox: TextBox, config: Config, rng: np.random.Generator) -> np.ndarray:
    """Applies a non-linear, wave-like distortion."""
    # --- Start of Fix ---
    # Guard against division by zero for textboxes with no height or width
    if textbox.height is None or textbox.height <= 0 or textbox.width is None or textbox.width <= 0:
        return points
    # --- End of Fix ---

    aug_config = config.textbox_distortion.warp_curl
    
    # Amplitudes are relative to the textbox size for consistent effect
    amp_x = textbox.height * sample_from_distribution(aug_config.amplitude_factor, rng)
    amp_y = textbox.width * sample_from_distribution(aug_config.amplitude_factor, rng)

    # Frequencies
    freq_x = (2 * np.pi / textbox.height) * sample_from_distribution(aug_config.frequency_factor_y, rng)
    freq_y = (2 * np.pi / textbox.width) * sample_from_distribution(aug_config.frequency_factor_x, rng)

    phase_x = sample_from_distribution(aug_config.phase_x, rng)
    phase_y = sample_from_distribution(aug_config.phase_y, rng)

    x, y = points[:, 0], points[:, 1]
    
    new_x = x + amp_x * np.sin(freq_x * y + phase_x)
    new_y = y + amp_y * np.sin(freq_y * x + phase_y)
    
    points[:, 0] = new_x
    points[:, 1] = new_y
    return points

@register_augmentation("linear_crease")
def apply_linear_crease(points: np.ndarray, textbox: TextBox, config: Config, rng: np.random.Generator) -> np.ndarray:
    """
    Applies one or more sequential Gaussian "creases" to the textbox,
    creating a localized dent/trough to simulate a finite fold in the paper.
    """
    if textbox.height is None or textbox.height <= 0 or textbox.width is None or textbox.width <= 0:
        return points

    aug_config = config.textbox_distortion.linear_crease
    num_creases = sample_from_distribution(aug_config.num_creases, rng)

    current_points = points.copy()

    for _ in range(num_creases):
        # Sample all parameters for this crease
        strength_factor = sample_from_distribution(aug_config.strength, rng)
        angle_deg = sample_from_distribution(aug_config.angle_deg, rng)
        position_factor = sample_from_distribution(aug_config.position_factor, rng)
        width_factor = sample_from_distribution(aug_config.crease_width_factor, rng)
        center_x_factor = sample_from_distribution(aug_config.crease_center_x_factor, rng)
        length_factor = sample_from_distribution(aug_config.crease_length_factor, rng)

        # 1. Define the crease line
        angle_rad = np.deg2rad(angle_deg)
        y_intercept = (textbox.height / 2) * position_factor

        A = np.sin(angle_rad)
        B = -np.cos(angle_rad)
        C = y_intercept * np.cos(angle_rad)

        px, py = current_points[:, 0], current_points[:, 1]

        # 2. Calculate perpendicular distance for vertical falloff
        distances_perp = A * px + B * py + C
        sigma_perp = textbox.height * width_factor
        if sigma_perp < 1e-6: continue
        vertical_falloff = np.exp(-(distances_perp**2) / (2 * sigma_perp**2))

        # --- NEW LOGIC: Calculate horizontal falloff for crease length ---
        # Define the horizontal center of the crease
        center_x = (textbox.width / 2) * center_x_factor
        
        # Calculate horizontal distance from each point to the crease's center
        distances_para = px - center_x
        
        # Sigma for the horizontal Gaussian defines the crease length
        sigma_para = textbox.width * length_factor
        if sigma_para < 1e-6: continue
        horizontal_falloff = np.exp(-(distances_para**2) / (2 * sigma_para**2))
        # --- END NEW LOGIC ---
        
        # 3. Combine falloffs and calculate total displacement
        total_falloff = vertical_falloff * horizontal_falloff
        max_displacement = textbox.height * strength_factor
        displacement_magnitude = max_displacement * total_falloff

        # 4. Calculate the displacement vector (always perpendicular to the crease)
        disp_x = displacement_magnitude * A
        disp_y = displacement_magnitude * B

        # 5. Apply the displacement
        current_points[:, 0] += disp_x
        current_points[:, 1] += disp_y

    return current_points