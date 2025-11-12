import numpy as np
import copy
from typing import List, Tuple, Optional

from manuscript_generator.core.textbox import TextBox
from manuscript_generator.configs.base_config import Config
from manuscript_generator.utils.distribution_sampler import sample_from_distribution
from manuscript_generator.utils.geometry import get_rotation_matrix

def _re_index_labels(points: np.ndarray, line_ids: np.ndarray, sub_box_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes a subset of points and their labels and re-indexes the line IDs to be zero-based
    and contiguous. The sub_box_ids are passed through but not modified.
    """
    if points.shape[0] == 0:
        return points, line_ids, sub_box_ids
    
    unique_original_ids = np.unique(line_ids)
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_original_ids)}
    
    new_line_ids = np.array([id_map[old_id] for old_id in line_ids], dtype=int)
    return points, new_line_ids, sub_box_ids


def _apply_split_augmentations(textbox: TextBox, split_method: str, config: Config, rng: np.random.Generator):
    """Applies augmentations to a split textbox, operating on all three parallel arrays."""
    aug_config = config.textbox_splitting.augmentation
    if not aug_config.enabled or textbox.points_local is None or textbox.points_local.shape[0] == 0:
        return

    # 1. Line Reduction
    reduction_ratio = sample_from_distribution(aug_config.line_reduction_ratio, rng)
    if reduction_ratio > 0:
        unique_lines = np.unique(textbox.line_ids_local)
        num_lines_to_drop = min(int(len(unique_lines) * reduction_ratio), len(unique_lines) - 1)
        if num_lines_to_drop > 0:
            line_positions = sorted([(np.mean(textbox.points_local[textbox.line_ids_local == lid, 1]), lid) for lid in unique_lines], key=lambda item: item[0], reverse=True)
            sorted_line_ids = [lid for _, lid in line_positions]
            lines_to_drop = sorted_line_ids[:num_lines_to_drop] if rng.random() < 0.5 else sorted_line_ids[-num_lines_to_drop:]
            keep_mask = ~np.isin(textbox.line_ids_local, lines_to_drop)
            textbox.points_local, textbox.line_ids_local, textbox.sub_box_ids_local = (arr[keep_mask] for arr in [textbox.points_local, textbox.line_ids_local, textbox.sub_box_ids_local])
            textbox.points_local, textbox.line_ids_local, textbox.sub_box_ids_local = _re_index_labels(
                textbox.points_local, textbox.line_ids_local, textbox.sub_box_ids_local)

    # 2. Font Size Variation
    font_factor = sample_from_distribution(aug_config.font_size_factor, rng)
    textbox.points_local[:, 2] *= font_factor

    # 3. Line Cutting
    line_cut_prob = sample_from_distribution(aug_config.line_cut_short_prob, rng)
    if rng.random() < line_cut_prob and textbox.points_local.shape[0] > 0:
        indices_to_keep = np.ones(textbox.points_local.shape[0], dtype=bool)
        for line_id in np.unique(textbox.line_ids_local):
            if rng.random() < line_cut_prob:
                line_mask = (textbox.line_ids_local == line_id)
                if not np.any(line_mask): continue
                line_points_x = textbox.points_local[line_mask, 0]
                min_x, max_x = np.min(line_points_x), np.max(line_points_x)
                if (line_width := max_x - min_x) < 1e-6: continue
                cut_amount = line_width * sample_from_distribution(aug_config.line_cut_short_factor, rng)
                cut_from_right = rng.random() < 0.5
                cutoff_val = max_x - cut_amount if cut_from_right else min_x + cut_amount
                points_to_cut_mask = (line_points_x > cutoff_val) if cut_from_right else (line_points_x < cutoff_val)
                original_indices = np.where(line_mask)[0]
                indices_to_cut = original_indices[points_to_cut_mask]
                indices_to_keep[indices_to_cut] = False
        textbox.points_local, textbox.line_ids_local, textbox.sub_box_ids_local = (arr[indices_to_keep] for arr in [textbox.points_local, textbox.line_ids_local, textbox.sub_box_ids_local])

    # --- NEW: 4. Geometric Stretching and Scaling ---
    # Horizontal Stretch (Character Spacing)
    stretch_x = sample_from_distribution(aug_config.stretch_x_factor, rng)
    if abs(1.0 - stretch_x) > 1e-6:
        for line_id in np.unique(textbox.line_ids_local):
            line_mask = (textbox.line_ids_local == line_id)
            if not np.any(line_mask): continue
            line_x_coords = textbox.points_local[line_mask, 0]
            start_x = np.min(line_x_coords)
            relative_x = line_x_coords - start_x
            textbox.points_local[line_mask, 0] = start_x + (relative_x * stretch_x)

    # Vertical Stretch (Line Spacing)
    stretch_y = sample_from_distribution(aug_config.stretch_y_factor, rng)
    if abs(1.0 - stretch_y) > 1e-6 and len(unique_lines := np.unique(textbox.line_ids_local)) > 1:
        line_positions = sorted([(np.mean(textbox.points_local[textbox.line_ids_local == lid, 1]), lid) for lid in unique_lines], key=lambda item: item[0], reverse=True)
        top_line_y_avg, _ = line_positions[0]
        for avg_y, line_id in line_positions[1:]:
            line_mask = (textbox.line_ids_local == line_id)
            relative_dist = avg_y - top_line_y_avg
            new_dist = relative_dist * stretch_y
            delta_y = new_dist - relative_dist
            textbox.points_local[line_mask, 1] += delta_y
            
    # Uniform Scaling
    scale = sample_from_distribution(aug_config.uniform_scale_factor, rng)
    if abs(1.0 - scale) > 1e-6:
        textbox.points_local[:, :2] *= scale

    # 5. Local Rotation
    angle_deg = sample_from_distribution(aug_config.rotation_deg, rng)
    if abs(angle_deg) > 1e-3:
        rot_matrix = get_rotation_matrix(angle_deg)
        textbox.points_local[:, :2] = textbox.points_local[:, :2] @ rot_matrix.T
    
    # 6. Conditional Shift or Crop-Specific Scale
    if split_method == "internal_crop":
        scale_factor = sample_from_distribution(aug_config.internal_crop_scale_factor, rng)
        textbox.points_local[:, :2] *= scale_factor
    elif split_method == "horizontal":
        shift_x = (textbox.width or 1.0) * sample_from_distribution(aug_config.horizontal_split_shift_x_factor, rng)
        shift_y = (textbox.height or 1.0) * sample_from_distribution(aug_config.horizontal_split_shift_y_factor, rng)
        textbox.points_local[:, 0] += shift_x; textbox.points_local[:, 1] += shift_y
    elif split_method == "vertical":
        shift_x = (textbox.width or 1.0) * sample_from_distribution(aug_config.vertical_split_shift_x_factor, rng)
        shift_y = (textbox.height or 1.0) * sample_from_distribution(aug_config.vertical_split_shift_y_factor, rng)
        textbox.points_local[:, 0] += shift_x; textbox.points_local[:, 1] += shift_y
        
    # 7. Final Recalculation of Bounding Box
    if textbox.points_local.shape[0] > 0:
        min_coords = np.min(textbox.points_local[:, :2], axis=0)
        max_coords = np.max(textbox.points_local[:, :2], axis=0)
        textbox.width = max_coords[0] - min_coords[0]
        textbox.height = max_coords[1] - min_coords[1]


def _create_split_textbox(original_textbox: TextBox, mask: np.ndarray) -> Optional[TextBox]:
    """Helper to create a new textbox from a subset of points, now including sub_box_ids."""
    if not np.any(mask):
        return None
        
    new_box = TextBox(box_type=original_textbox.box_type)
    
    # --- MODIFIED: Propagate all three arrays ---
    points, line_ids, sub_box_ids = _re_index_labels(
        original_textbox.points_local[mask],
        original_textbox.line_ids_local[mask],
        original_textbox.sub_box_ids_local[mask] # <-- Pass through
    )
    new_box.points_local = points
    new_box.line_ids_local = line_ids
    new_box.sub_box_ids_local = sub_box_ids # <-- Assign

    if new_box.points_local.shape[0] > 0:
        min_coords = np.min(new_box.points_local[:, :2], axis=0)
        max_coords = np.max(new_box.points_local[:, :2], axis=0)
        new_box.width = max_coords[0] - min_coords[0]
        new_box.height = max_coords[1] - min_coords[1]
    else:
        new_box.width = 0; new_box.height = 0
        
    return new_box

def _split_horizontal(textbox: TextBox, rng: np.random.Generator) -> List[TextBox]:
    """Splits a textbox into top and bottom halves."""
    y_coords = textbox.points_local[:, 1]
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    # --- MODIFIED: Constrain split point ---
    y_range = max_y - min_y
    split_point = rng.uniform(min_y + y_range * 0.15, max_y - y_range * 0.15)
    
    mask_top = y_coords >= split_point
    mask_bottom = y_coords < split_point
    
    box1 = _create_split_textbox(textbox, mask_top)
    box2 = _create_split_textbox(textbox, mask_bottom)
    
    return [b for b in [box1, box2] if b]

def _split_vertical(textbox: TextBox, rng: np.random.Generator) -> List[TextBox]:
    """Splits a textbox into left and right halves."""
    x_coords = textbox.points_local[:, 0]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    # --- MODIFIED: Constrain split point ---
    x_range = max_x - min_x
    split_point = rng.uniform(min_x + x_range * 0.15, max_x - x_range * 0.15)
    
    mask_left = x_coords < split_point
    mask_right = x_coords >= split_point
    
    box1 = _create_split_textbox(textbox, mask_left)
    box2 = _create_split_textbox(textbox, mask_right)
    
    return [b for b in [box1, box2] if b]

def _split_internal_crop(textbox: TextBox, rng: np.random.Generator) -> List[TextBox]:
    # (This function remains unchanged)
    x, y = textbox.points_local[:, 0], textbox.points_local[:, 1]
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    
    x_margin = (max_x - min_x) * rng.uniform(0.1, 0.3)
    y_margin = (max_y - min_y) * rng.uniform(0.1, 0.3)
    
    inner_min_x, inner_max_x = min_x + x_margin, max_x - x_margin
    inner_min_y, inner_max_y = min_y + y_margin, max_y - y_margin
    
    mask_inner = (x >= inner_min_x) & (x <= inner_max_x) & (y >= inner_min_y) & (y <= inner_max_y)
    
    box_inner = _create_split_textbox(textbox, mask_inner)
    box_outer = _create_split_textbox(textbox, ~mask_inner)

    return [b for b in [box_inner, box_outer] if b]


def attempt_textbox_split(textbox: TextBox, config: Config, rng: np.random.Generator) -> List[TextBox]:
    """
    Main function for this module. Decides whether to split a textbox,
    performs the split, and augments one of the results.
    """
    split_config = config.textbox_splitting
    if not split_config.enabled or rng.random() > split_config.probability:
        return [textbox]

    if textbox.points_local is None or textbox.points_local.shape[0] < 10:
        return [textbox]

    # --- MODIFIED: Bias split method based on aspect ratio ---
    base_probs = split_config.split_method_probabilities
    methods = list(base_probs.keys())
    weights = np.array(list(base_probs.values()))

    if textbox.height and textbox.width and "horizontal" in methods and "vertical" in methods:
        h_idx = methods.index("horizontal")
        v_idx = methods.index("vertical")
        bias = min(abs(split_config.aspect_ratio_bias), 0.5) # Clamp bias

        if textbox.height > textbox.width: # Taller than wide -> bias to horizontal split
            weights[h_idx] += bias
            weights[v_idx] -= bias
        elif textbox.width > textbox.height: # Wider than tall -> bias to vertical split
            weights[v_idx] += bias
            weights[h_idx] -= bias
        
        weights = np.maximum(weights, 0) # Ensure no negative probabilities
        weights /= np.sum(weights) # Re-normalize

    chosen_method = rng.choice(methods, p=weights)
    # --- END MODIFICATION ---

    split_boxes = []
    if chosen_method == "horizontal":
        split_boxes = _split_horizontal(textbox, rng)
    elif chosen_method == "vertical":
        split_boxes = _split_vertical(textbox, rng)
    elif chosen_method == "internal_crop":
        split_boxes = _split_internal_crop(textbox, rng)
    
    if len(split_boxes) == 2:
        box_to_augment_idx = rng.choice([0, 1])
        box_to_augment = split_boxes[box_to_augment_idx]
        
        # --- MODIFIED: For internal crop, always augment the INNER box ---
        if chosen_method == "internal_crop":
            # Heuristic: the inner box is likely to have fewer points
            if split_boxes[0].points_local.shape[0] < split_boxes[1].points_local.shape[0]:
                box_to_augment = split_boxes[0]
            else:
                box_to_augment = split_boxes[1]

        _apply_split_augmentations(box_to_augment, chosen_method, config, rng)
        return split_boxes
    
    return [textbox]