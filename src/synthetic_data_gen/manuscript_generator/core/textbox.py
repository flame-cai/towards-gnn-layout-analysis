# /manuscript_generator/core/textbox.py

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from shapely.geometry import Polygon

from manuscript_generator.configs.base_config import Config
from manuscript_generator.core.common import Point, Word, TextLine, TextBoxType, TextAlignment
from manuscript_generator.utils.distribution_sampler import sample_from_distribution
from manuscript_generator.utils.geometry import get_convex_hull, get_rotation_matrix, apply_transform
from manuscript_generator.core.registry import AUGMENTATIONS

@dataclass
class TextBox:
    """
    Represents a textbox. It's initially generated in local coordinates,
    then consolidated into NumPy arrays for efficient augmentation and placement.
    A single TextBox can contain sub-components (main text vs. interlinear glosses)
    which will be assigned unique labels at the page finalization stage.
    """
    box_type: TextBoxType
    text_lines: List[TextLine] = field(default_factory=list)
    
    # Global properties (applied in Phase 3)
    position: Tuple[float, float] = (0, 0)
    orientation_deg: float = 0.0
    
    # Internal state for processing
    points_local: Optional[np.ndarray] = None
    line_ids_local: Optional[np.ndarray] = None
    # --- NEW: Array to track sub-components (0=main, 1=gloss1, 2=gloss2, etc.) ---
    sub_box_ids_local: Optional[np.ndarray] = None
    
    width: Optional[float] = None
    height: Optional[float] = None
    
    # Final state in global coordinates
    points_global: Optional[np.ndarray] = None
    hull_global: Optional[Polygon] = None

    def consolidate_to_numpy(self):
        """
        Converts the object-based text line structure into parallel NumPy arrays.
        This now includes a `sub_box_ids_local` array to differentiate main text from glosses.
        """
        points_list = []
        line_ids_list = []
        sub_box_ids_list = [] # <-- NEW
        
        current_line_id = 0
        current_sub_box_id = 0 # This will be 0 for main text, and increment for each gloss
        
        for text_line in self.text_lines:
            # --- 1. Process the main line of text (Sub-box ID = 0) ---
            if text_line.words:
                main_text_sub_box_id = 0
                for word in text_line.words:
                    for point in word.points:
                        points_list.append([point.x, point.y, point.font_size])
                        line_ids_list.append(current_line_id)
                        sub_box_ids_list.append(main_text_sub_box_id)
                current_line_id += 1

            # --- 2. Process the gloss above, if it exists ---
            if text_line.interlinear_gloss_above:
                current_sub_box_id += 1 # This gloss gets a NEW, unique sub-box ID
                for word in text_line.interlinear_gloss_above:
                    for point in word.points:
                        points_list.append([point.x, point.y, point.font_size])
                        line_ids_list.append(current_line_id)
                        sub_box_ids_list.append(current_sub_box_id)
                current_line_id += 1

            # --- 3. Process the gloss below, if it exists ---
            if text_line.interlinear_gloss_below:
                current_sub_box_id += 1 # This gloss also gets its own unique ID
                for word in text_line.interlinear_gloss_below:
                    for point in word.points:
                        points_list.append([point.x, point.y, point.font_size])
                        line_ids_list.append(current_line_id)
                        sub_box_ids_list.append(current_sub_box_id)
                current_line_id += 1

        if not points_list:
            self.points_local = np.empty((0, 3))
            self.line_ids_local = np.empty((0,), dtype=int)
            self.sub_box_ids_local = np.empty((0,), dtype=int) # <-- NEW
            self.width = 0
            self.height = 0
            return

        self.points_local = np.array(points_list, dtype=np.float64)
        self.line_ids_local = np.array(line_ids_list, dtype=int)
        self.sub_box_ids_local = np.array(sub_box_ids_list, dtype=int) # <-- NEW

        min_coords = np.min(self.points_local[:, :2], axis=0)
        max_coords = np.max(self.points_local[:, :2], axis=0)
        self.width = max_coords[0] - min_coords[0]
        self.height = max_coords[1] - min_coords[1]
        
        center_offset = (min_coords + max_coords) / 2.0
        self.points_local[:, :2] -= center_offset
        
        # --- Assert all three arrays are synchronized ---
        assert self.points_local.shape[0] == self.line_ids_local.shape[0]
        assert self.points_local.shape[0] == self.sub_box_ids_local.shape[0]
        assert self.points_local.shape[1] == 3


    def apply_augmentations(self, config: Config, rng: np.random.Generator):
        """Applies Phase 1 and 2 augmentations to the local point cloud."""
        # --- Phase 1: Content & Micro-Variations ---
        for aug_name in ["font_size_variation", "point_level_jitter", "congestion_jitter"]:
            aug_config = getattr(config.textbox_content, aug_name)
            if aug_config['enabled']:
                self.points_local = AUGMENTATIONS[aug_name](self.points_local, self.line_ids_local, config, rng)

        # --- Phase 2: Geometric Distortion ---
        distort_config = config.textbox_distortion
        distortions = list(distort_config.model_dump().keys())
        rng.shuffle(distortions)

        for aug_name in distortions:
            aug_config = getattr(distort_config, aug_name)
            if aug_config.enabled and rng.random() < aug_config.probability:
                self.points_local = AUGMENTATIONS[aug_name](self.points_local, self, config, rng)
    
    def transform_to_global(self):
        """
        Applies rotation and translation to move points from local to global
        page coordinates. Must be called after setting `position` and `orientation_deg`.
        """
        assert self.points_local is not None, "Must consolidate before transforming."
        
        rot_matrix = get_rotation_matrix(self.orientation_deg)
        rotated_points = apply_transform(self.points_local, rot_matrix)
        
        rotated_points[:, :2] += self.position
        self.points_global = rotated_points

        self.hull_global = get_convex_hull(self.points_global)

        
# --- MODIFICATION: Function signature now accepts main_char_spacing ---
def _create_one_gloss_line(
    base_y: float,
    y_sign: int,
    main_line_width: float,
    main_char_spacing: float, # <-- ADD THIS
    line_spacing: float,
    base_font_size: float,
    content_config: Config,
    rng: np.random.Generator
) -> List[Word]:
    """Generates the words for a single gloss line, placed at a random horizontal offset."""
    gloss_config = content_config.interlinear_gloss
    
    gloss_font_size = base_font_size * sample_from_distribution(gloss_config.font_size_factor, rng)

    # --- MODIFIED: Calculate gloss character spacing relative to the main line's ---
    spacing_multiplier = sample_from_distribution(gloss_config.character_spacing_multiplier, rng)
    gloss_char_spacing = main_char_spacing * spacing_multiplier
    # --- END MODIFICATION ---
    
    gloss_word_spacing = gloss_char_spacing * sample_from_distribution(content_config.word_spacing_factor, rng)
    
    y_offset_factor = sample_from_distribution(gloss_config.vertical_offset_factor, rng)
    gloss_y = base_y + y_sign * (line_spacing * y_offset_factor)

    gloss_words = []
    gloss_current_x = 0
    gloss_words_per_line = sample_from_distribution(gloss_config.words_per_line, rng)
    for _ in range(gloss_words_per_line):
        chars_in_word_gloss = sample_from_distribution(gloss_config.chars_per_word, rng)
        points = [Point(x=gloss_current_x + i * gloss_char_spacing, y=gloss_y, font_size=gloss_font_size) for i in range(chars_in_word_gloss)]
        gloss_current_x += (chars_in_word_gloss * gloss_char_spacing) + gloss_word_spacing
        gloss_words.append(Word(points=points))
        
    if not gloss_words:
        return []
        
    gloss_width = gloss_current_x - gloss_word_spacing
    slack_space = main_line_width - gloss_width
    
    horizontal_offset = 0
    if slack_space > 0:
        horizontal_offset = rng.uniform(0, slack_space)

    if horizontal_offset > 0:
        for word in gloss_words:
            for point in word.points:
                point.x += horizontal_offset
        
    return gloss_words

def _create_text_lines(box_config: dict, content_config: Config, rng: np.random.Generator) -> Tuple[List[TextLine], float, float]:
    """Helper to generate the raw text lines for a textbox, including interlinear glosses."""
    lines_per_box = sample_from_distribution(box_config.lines_per_box, rng)
    base_font_size = sample_from_distribution(box_config.font_size, rng)

    char_spacing = base_font_size * sample_from_distribution(content_config.character_spacing_factor, rng)
    word_spacing = char_spacing * sample_from_distribution(content_config.word_spacing_factor, rng)
    line_spacing = char_spacing * sample_from_distribution(content_config.line_spacing_factor, rng)
    line_break_probability = sample_from_distribution(content_config.line_break_probability, rng)

    text_lines = []
    max_line_width = 0
    current_y = 0
    
    # --- MODIFIED: Implement the "sample-once-then-vary" pattern ---
    # 1. Sample the base properties for the entire textbox once.
    base_words_per_line = sample_from_distribution(box_config.words_per_line, rng)
    base_chars_per_word = sample_from_distribution(box_config.chars_per_word, rng)
    variation_factor = sample_from_distribution(content_config.line_length_variation.variation_factor, rng)
    # --- END MODIFICATION ---

    has_gloss_prob = getattr(box_config, 'interlinear_gloss_probability', 0)
    
    for _ in range(lines_per_box):
        # --- MODIFIED: Apply slight variation to the base values for each line ---
        # Calculate the allowed character deviation for this line
        char_variation = int(base_chars_per_word * variation_factor)
        
        # Apply a random delta within the allowed variation
        # The +1 is needed because rng.integers has an exclusive upper bound
        random_delta = rng.integers(-char_variation, char_variation + 1) if char_variation > 0 else 0
        
        # Ensure the final character count is at least 1
        final_chars_per_word = max(1, base_chars_per_word + random_delta)
        # --- END MODIFICATION ---

        # Generate the main line of text using the calculated `final_chars_per_word`
        current_x = 0
        main_words = []
        for _ in range(base_words_per_line): # Assuming words_per_line is constant
            points = []
            if rng.random() > line_break_probability:
                num_chars_in_word = final_chars_per_word
            else:
                num_chars_in_word = rng.integers(1, final_chars_per_word) if final_chars_per_word > 1 else 1

            for i in range(num_chars_in_word):
                points.append(Point(x=current_x + i * char_spacing, y=current_y, font_size=base_font_size))
            current_x += (num_chars_in_word * char_spacing) + word_spacing
            main_words.append(Word(points=points))
        
        line_width = current_x - word_spacing
        if line_width > max_line_width:
            max_line_width = line_width
            
        text_line = TextLine(words=main_words)

        # ... (The rest of the function, including gloss generation, remains unchanged) ...
        if rng.random() < has_gloss_prob:
            gloss_config = content_config.interlinear_gloss
            placement = sample_from_distribution(gloss_config.placement, rng)

            if placement in ["above", "both"]:
                gloss_above_words = _create_one_gloss_line(
                    current_y, y_sign=1, main_line_width=line_width, main_char_spacing=char_spacing,
                    line_spacing=line_spacing, base_font_size=base_font_size, content_config=content_config, rng=rng
                )
                text_line.interlinear_gloss_above = gloss_above_words

            if placement in ["below", "both"]:
                gloss_below_words = _create_one_gloss_line(
                    current_y, y_sign=-1, main_line_width=line_width, main_char_spacing=char_spacing,
                    line_spacing=line_spacing, base_font_size=base_font_size, content_config=content_config, rng=rng
                )
                text_line.interlinear_gloss_below = gloss_below_words

        text_lines.append(text_line)
        current_y -= line_spacing
        
    return text_lines, max_line_width, (abs(current_y) - line_spacing)

# ... (_apply_text_alignment and create_textbox are unchanged) ...
def _apply_text_alignment(text_lines: List[TextLine], alignment: TextAlignment, box_width: float, content_config: Config, rng: np.random.Generator):
    """Applies alignment to the text lines *in place*."""
    if alignment == TextAlignment.LEFT:
        return # Default is left-aligned

    for line in text_lines:
        if not line.words or not line.words[-1].points:
            continue
        
        # Calculate the natural width of the line
        first_point = line.words[0].points[0]
        last_point = line.words[-1].points[-1]
        line_width = last_point.x - first_point.x

        offset = 0
        if alignment == TextAlignment.RIGHT:
            offset = box_width - line_width
        elif alignment == TextAlignment.CENTER:
            offset = (box_width - line_width) / 2
        elif alignment == TextAlignment.JUSTIFY:
            # Handle main line justify
            if len(line.words) > 1:
                slack = box_width - line_width
                if slack > 0:
                    extra_space_per_gap = slack / (len(line.words) - 1)
                    cumulative_extra_space = 0
                    for i in range(1, len(line.words)):
                        cumulative_extra_space += extra_space_per_gap
                        for point in line.words[i].points:
                            point.x += cumulative_extra_space
            # Also apply alignment to glosses (non-justify)
            gloss_lines = [line.interlinear_gloss_above, line.interlinear_gloss_below]
            for gloss_line in gloss_lines:
                if gloss_line:
                    gloss_last_point = gloss_line[-1].points[-1]
                    gloss_width = gloss_last_point.x - gloss_line[0].points[0].x
                    gloss_offset = 0
                    if alignment == TextAlignment.RIGHT:
                        gloss_offset = box_width - gloss_width
                    elif alignment == TextAlignment.CENTER:
                        gloss_offset = (box_width - gloss_width) / 2
                    
                    if gloss_offset != 0:
                        for word in gloss_line:
                            for point in word.points:
                                point.x += gloss_offset
            continue # Justify logic is self-contained

        # Apply offset to all points in the line (and its glosses)
        all_words = list(line.words)
        if line.interlinear_gloss_above: all_words.extend(line.interlinear_gloss_above)
        if line.interlinear_gloss_below: all_words.extend(line.interlinear_gloss_below)

        for word in all_words:
            for point in word.points:
                point.x += offset

def create_textbox(box_type: TextBoxType, config: Config, rng: np.random.Generator) -> TextBox:
    """Factory function to create a fully formed TextBox object."""
    content_config = config.textbox_content
    box_specific_config = getattr(content_config, box_type.value)
    
    # 1. Generate text content (Phase 1 pre-cursor)
    text_lines, max_line_width, _ = _create_text_lines(box_specific_config, content_config, rng)
    
    # 2. Apply text alignment
    alignment = TextAlignment(sample_from_distribution(box_specific_config.alignment, rng))
    _apply_text_alignment(text_lines, alignment, max_line_width, content_config, rng)
    
    # 3. Create TextBox object and consolidate to numpy
    textbox = TextBox(box_type=box_type, text_lines=text_lines)
    textbox.consolidate_to_numpy()
    
    # 4. Apply Phase 1 and 2 augmentations
    if textbox.points_local.shape[0] > 0:
        textbox.apply_augmentations(config, rng)
        
    return textbox