# /manuscript_generator/core/page.py

import numpy as np
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

from manuscript_generator.core.textbox import TextBox
from manuscript_generator.configs.base_config import Config

@dataclass
class Page:
    """Represents a single generated manuscript page."""
    width: int
    height: int
    textboxes: List[TextBox] = field(default_factory=list)
    
    # Final data for output
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    # --- RENAMED: This now stores labels for sub-textboxes (main text, glosses) ---
    sub_textbox_labels: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))
    textline_labels: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))

    def finalize(self, config: Config, rng: np.random.Generator):
        """
        Combines all textboxes into final page-level arrays, generating globally unique
        labels for both sub-textboxes (main text vs glosses) and text lines.
        """
        if not self.textboxes:
            return

        all_points = []
        all_sub_textbox_labels = [] # <-- RENAMED
        all_textline_labels = []
        
        global_line_id_offset = 0
        global_textbox_id_offset = 0 # <-- NEW offset for sub-textbox labels

        # --- MODIFIED: The loop now manages two separate global ID counters ---
        for box in self.textboxes:
            if box.points_global is None or box.points_global.shape[0] == 0:
                continue
                
            all_points.append(box.points_global)
            
            # --- Generate globally unique sub-textbox labels ---
            if box.sub_box_ids_local is not None and box.sub_box_ids_local.size > 0:
                global_sub_box_ids = box.sub_box_ids_local + global_textbox_id_offset
                all_sub_textbox_labels.append(global_sub_box_ids)
                # Increment the global offset by the number of unique sub-boxes in this textbox
                global_textbox_id_offset += np.max(box.sub_box_ids_local) + 1
            else: # Fallback for safety
                all_sub_textbox_labels.append(np.full(box.points_global.shape[0], global_textbox_id_offset))
                global_textbox_id_offset += 1

            # --- Generate globally unique textline labels (logic is unchanged) ---
            if box.line_ids_local is not None and box.line_ids_local.size > 0:
                global_line_ids = box.line_ids_local + global_line_id_offset
                all_textline_labels.append(global_line_ids)
                global_line_id_offset += np.max(box.line_ids_local) + 1
            elif box.points_global.shape[0] > 0:
                all_textline_labels.append(np.full(box.points_global.shape[0], global_line_id_offset))
                global_line_id_offset += 1

        if not all_points:
            return
            
        self.points = np.vstack(all_points)
        self.sub_textbox_labels = np.concatenate(all_sub_textbox_labels) # <-- RENAMED
        self.textline_labels = np.concatenate(all_textline_labels)

        # Apply Phase 3 augmentations (e.g., Point Dropout)
        from manuscript_generator.core.registry import AUGMENTATIONS
        self.points, kept_indices = AUGMENTATIONS['point_dropout'](self.points, None, config, rng)
        
        # --- Apply the filter mask to BOTH label arrays ---
        self.sub_textbox_labels = self.sub_textbox_labels[kept_indices] # <-- RENAMED
        self.textline_labels = self.textline_labels[kept_indices]
        
    def save(self, output_dir: Path, sample_id: str):
        """Saves all generated data for this page to disk."""
        longest_dim = max(self.width, self.height)
        if longest_dim == 0: return

        points_normalized = self.points.copy()
        points_normalized[:, :2] /= longest_dim
        if self.height > 0:
            points_normalized[:, 2] /= longest_dim
        
        # --- Save all files ---
        np.savetxt(output_dir / f"{sample_id}_inputs_unnormalized.txt", self.points, fmt="%.2f %.2f %d")
        np.savetxt(output_dir / f"{sample_id}_inputs_normalized.txt", points_normalized, fmt="%.6f %.6f %.6f")
        # --- MODIFIED: Save the new sub-textbox labels to the correct file ---
        # Note: The output filename `_labels_textbox.txt` is kept for compatibility with downstream consumers.
        np.savetxt(output_dir / f"{sample_id}_labels_region.txt", self.sub_textbox_labels, fmt="%d")
        np.savetxt(output_dir / f"{sample_id}_labels_textline.txt", self.textline_labels, fmt="%d")
        with open(output_dir / f"{sample_id}_dims.txt", 'w') as f:
            f.write(f"{self.width} {self.height}")