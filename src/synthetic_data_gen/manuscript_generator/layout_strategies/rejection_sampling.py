# /manuscript_generator/layout_strategies/rejection_sampling.py

import numpy as np
from typing import List

from manuscript_generator.core.registry import register_layout
from manuscript_generator.core.page import Page
from manuscript_generator.core.textbox import create_textbox
from manuscript_generator.core.common import TextBoxType
from manuscript_generator.configs.base_config import Config
from manuscript_generator.utils.distribution_sampler import sample_from_distribution
from manuscript_generator.utils.geometry import check_overlap, get_rotation_matrix
from manuscript_generator.augmentations.phase3_5_splitting import attempt_textbox_split

@register_layout("rejection_sampling")
def generate_rejection_sampling_layout(config: Config, rng: np.random.Generator) -> List[Page]:
    """
    Generates a page layout by creating textboxes and placing them using rejection sampling
    to avoid overlaps. Includes logic to split textboxes and place the fragments closely.
    """
    # --- MODIFIED: Page dimension sampling logic ---
    page_config = config.page
    orientations = list(page_config.orientation_probabilities.keys())
    probs = list(page_config.orientation_probabilities.values())
    chosen_orientation = rng.choice(orientations, p=probs)

    if chosen_orientation == 'portrait':
        page_width = sample_from_distribution(page_config.short_side, rng)
        page_height = sample_from_distribution(page_config.long_side, rng)
    elif chosen_orientation == 'landscape':
        page_width = sample_from_distribution(page_config.long_side, rng)
        page_height = sample_from_distribution(page_config.short_side, rng)
    else:  # 'square'
        side = sample_from_distribution(page_config.square_side, rng)
        page_width = side
        page_height = side
    # --- END MODIFICATION ---

    page = Page(width=page_width, height=page_height)

    num_textboxes = sample_from_distribution(config.rejection_sampling.num_textboxes, rng)
    box_types_config = config.rejection_sampling.textbox_type_probabilities
    types_to_generate = rng.choice(
        list(box_types_config.keys()),
        size=num_textboxes,
        p=list(box_types_config.values())
    )
    types_to_generate = np.insert(types_to_generate, 0, "main_text")
    
    for i, box_type_str in enumerate(types_to_generate):
        box_type = TextBoxType(box_type_str)
    
        max_box_attempts = sample_from_distribution(config.rejection_sampling.max_box_generation_attempts, rng)
        
        for _ in range(max_box_attempts):
            # 1. Create a textbox as before
            textbox = create_textbox(box_type, config, rng)
            if textbox.points_local is None or textbox.points_local.shape[0] < 3:
                continue

            # --- NEW: Attempt to split the textbox ---
            textboxes_to_place = attempt_textbox_split(textbox, config, rng)
            # --- END NEW ---

            max_placement_attempts = sample_from_distribution(config.rejection_sampling.max_placement_attempts, rng)
            
            placed_boxes_for_this_attempt = []
            
            # --- MODIFIED: Placement logic to handle one or two boxes ---
            is_split_attempt = len(textboxes_to_place) > 1
            anchor_box = textboxes_to_place[0]

            # Try to place the first (anchor) box
            anchor_placed = False
            for _ in range(max_placement_attempts):
                anchor_box.position = (rng.uniform(0, page.width), rng.uniform(0, page.height))
                # (The orientation logic remains the same)
                orientation_choice = sample_from_distribution(config.page_augmentations.orientation_deg, rng)
                final_orientation = 0.0
                if orientation_choice == "other":
                    base_angle = rng.choice([0, 90, -90])
                    offset = sample_from_distribution(config.page_augmentations.orientation_other_range, rng)
                    final_orientation = base_angle + offset
                else:
                    final_orientation = float(orientation_choice)
                anchor_box.orientation_deg = final_orientation
                anchor_box.transform_to_global()

                if anchor_box.hull_global is None: continue
                min_x, min_y, max_x, max_y = anchor_box.hull_global.bounds
                if not (0 <= min_x < page_width and 0 <= min_y < page_height and
                        0 < max_x <= page_width and 0 < max_y <= page_height):
                    continue
                
                if not any(check_overlap(anchor_box.hull_global, b.hull_global) for b in page.textboxes):
                    anchor_placed = True
                    break
            
            if not anchor_placed:
                continue # Failed to place even the first box, so try generating a new one

            placed_boxes_for_this_attempt.append(anchor_box)
            
            # If it was a split, now try to place the second box near the anchor
            all_placed_successfully = True
            if is_split_attempt:
                sibling_box = textboxes_to_place[1]
                sibling_placed = False
                # Try to place the sibling very close to the anchor
                for _ in range(max_placement_attempts):
                    # Sample position relative to the anchor's position
                    pos_offset_x = (anchor_box.width or 10) * rng.uniform(-0.5, 0.5)
                    pos_offset_y = (anchor_box.height or 10) * rng.uniform(-0.5, 0.5)
                    sibling_box.position = (anchor_box.position[0] + pos_offset_x, anchor_box.position[1] + pos_offset_y)
                    
                    # Give it a slightly different orientation
                    sibling_box.orientation_deg = anchor_box.orientation_deg + rng.uniform(-5, 5)
                    sibling_box.transform_to_global()

                    if sibling_box.hull_global is None: continue
                    min_x, min_y, max_x, max_y = sibling_box.hull_global.bounds
                    if not (0 <= min_x < page_width and 0 <= min_y < page_height and
                            0 < max_x <= page_width and 0 < max_y <= page_height):
                        continue

                    # Check for overlap with existing page boxes AND the anchor we just placed
                    if not any(check_overlap(sibling_box.hull_global, b.hull_global) for b in page.textboxes + [anchor_box]):
                        sibling_placed = True
                        break

                if sibling_placed:
                    placed_boxes_for_this_attempt.append(sibling_box)
                else:
                    all_placed_successfully = False # Failed to place the sibling, so we abort this entire attempt
            
            # --- Final commit or rollback ---
            if all_placed_successfully:
                page.textboxes.extend(placed_boxes_for_this_attempt)
                break # Success, move to generating the next textbox
            # If not successful, the loop continues to the next box generation attempt
            # --- END MODIFIED ---
    
    return [page]