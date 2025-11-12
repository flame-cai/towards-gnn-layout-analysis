# manuscript_generator/augmentations/phase4_page.py

import numpy as np
import logging

from manuscript_generator.core.registry import register_augmentation
from manuscript_generator.utils.distribution_sampler import sample_from_distribution
from manuscript_generator.utils.geometry import get_rotation_matrix
from ..configs.augmentation_config import AugmentationConfig

@register_augmentation("page_rotation")
def apply_page_rotation(points: np.ndarray, page_dims: dict, config: AugmentationConfig, rng: np.random.Generator) -> np.ndarray:
    """Rotates the entire page's point cloud around its center."""
    aug_config = config.phase4_page.page_rotation
    angle_deg = sample_from_distribution(aug_config.rotation_deg, rng)
    
    if abs(angle_deg) < 1e-3:
        return points

    logging.debug(f"Applying page rotation of {angle_deg:.2f} degrees.")

    center_x = page_dims['width'] / 2.0
    center_y = page_dims['height'] / 2.0
    
    # Copy points to avoid modifying the original array in place
    aug_points = points.copy()
    
    # 1. Translate points to origin
    aug_points[:, 0] -= center_x
    aug_points[:, 1] -= center_y

    # 2. Rotate
    rot_matrix = get_rotation_matrix(angle_deg)
    rotated_coords = aug_points[:, :2] @ rot_matrix.T
    
    # 3. Translate back
    aug_points[:, 0] = rotated_coords[:, 0] + center_x
    aug_points[:, 1] = rotated_coords[:, 1] + center_y
    
    return aug_points

@register_augmentation("page_translation")
def apply_page_translation(points: np.ndarray, page_dims: dict, config: AugmentationConfig, rng: np.random.Generator) -> np.ndarray:
    """Translates the entire page's point cloud by a random offset."""
    aug_config = config.phase4_page.page_translation
    
    dx_factor = sample_from_distribution(aug_config.translate_x_factor, rng)
    dy_factor = sample_from_distribution(aug_config.translate_y_factor, rng)
    
    dx = page_dims['width'] * dx_factor
    dy = page_dims['height'] * dy_factor

    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return points
    
    logging.debug(f"Applying page translation of (dx={dx:.2f}, dy={dy:.2f}).")
    
    aug_points = points.copy()
    aug_points[:, 0] += dx
    aug_points[:, 1] += dy
    
    return aug_points

@register_augmentation("page_mirror")
def apply_page_mirror(points: np.ndarray, page_dims: dict, config: AugmentationConfig, rng: np.random.Generator) -> np.ndarray:
    """Applies horizontal or vertical mirroring to the page's point cloud."""
    aug_config = config.phase4_page.page_mirror
    aug_points = points.copy()
    
    # Horizontal Mirror
    if rng.random() < aug_config.horizontal_prob:
        center_x = page_dims['width'] / 2.0
        logging.debug("Applying horizontal page mirror.")
        aug_points[:, 0] = 2 * center_x - aug_points[:, 0]

    # Vertical Mirror
    if rng.random() < aug_config.vertical_prob:
        center_y = page_dims['height'] / 2.0
        logging.debug("Applying vertical page mirror.")
        aug_points[:, 1] = 2 * center_y - aug_points[:, 1]
        
    return aug_points