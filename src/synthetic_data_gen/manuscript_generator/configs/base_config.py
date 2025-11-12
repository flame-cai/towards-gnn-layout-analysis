# /manuscript_generator/configs/base_config.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Union, Literal

# ... (keep all existing class definitions: Distribution, UniformInt, etc.) ...
class Distribution(BaseModel):
    dist: str

class UniformInt(Distribution):
    dist: Literal["uniform_int"]
    min: int
    max: int

class UniformFloat(Distribution):
    dist: Literal["uniform_float"]
    min: float
    max: float

class Normal(Distribution):
    dist: Literal["normal"]
    mean: float
    std: float

class Constant(Distribution):
    dist: Literal["constant"]
    value: Any

class Choice(Distribution):
    dist: Literal["choice"]
    choices: List[Any]
    weights: List[float] = None

AnyDist = Union[UniformInt, UniformFloat, Normal, Constant, Choice]

class GenerationConfig(BaseModel):
    num_samples: int
    num_workers: int
    base_seed: int
    output_dir: str
    dry_run_num_samples: int

class VisualizationConfig(BaseModel):
    enabled: bool
    render_on_dry_run_only: bool
    coloring: Literal["textbox", "textline"]
    point_size_multiplier: float

class PageConfig(BaseModel):
    # Old `width` and `height` are replaced with these more descriptive fields.
    orientation_probabilities: Dict[str, float]
    long_side: AnyDist
    short_side: AnyDist
    square_side: AnyDist
    layout_strategy: AnyDist

class RejectionSamplingConfig(BaseModel):
    num_textboxes: AnyDist
    max_placement_attempts: AnyDist
    max_box_generation_attempts: AnyDist
    textbox_type_probabilities: Dict[str, float]

class GridLayoutConfig(BaseModel):
    rows: AnyDist
    cols: AnyDist
    spacing: AnyDist
    augmentations: List[str]

class TextBoxContentParams(BaseModel):
    font_size: AnyDist
    lines_per_box: AnyDist
    words_per_line: AnyDist
    alignment: AnyDist
    interlinear_gloss_probability: float = 0.0 # Add with a default value
    chars_per_word: AnyDist


class InterlinearGlossConfig(BaseModel):
    font_size_factor: AnyDist
    words_per_line: AnyDist
    vertical_offset_factor: AnyDist
    chars_per_word: AnyDist
    placement: AnyDist
    character_spacing_multiplier: AnyDist

class LineLengthVariationConfig(BaseModel):
    variation_factor: AnyDist

class TextBoxContentConfig(BaseModel):
    main_text: TextBoxContentParams
    marginalia: TextBoxContentParams
    page_number: TextBoxContentParams
    interlinear_gloss: InterlinearGlossConfig
    character_spacing_factor: AnyDist
    word_spacing_factor: AnyDist
    line_spacing_factor: AnyDist
    line_break_probability: AnyDist
    line_length_variation: LineLengthVariationConfig
    font_size_variation: Dict[str, Any]
    point_level_jitter: Dict[str, Any]
    congestion_jitter: Dict[str, Any]
    interlinear_gloss: InterlinearGlossConfig
    

class DistortionAugmentationConfig(BaseModel):
    enabled: bool
    probability: float

class ShearConfig(DistortionAugmentationConfig):
    shear_factor_x: AnyDist
    shear_factor_y: AnyDist

class StretchConfig(DistortionAugmentationConfig):
    stretch_factor_x: AnyDist
    stretch_factor_y: AnyDist

class WarpCurlConfig(DistortionAugmentationConfig):
    amplitude_factor: AnyDist
    frequency_factor_x: AnyDist
    frequency_factor_y: AnyDist
    phase_x: AnyDist
    phase_y: AnyDist

# --- NEW: Add the config model for our new augmentation ---
class LinearCreaseConfig(DistortionAugmentationConfig):
    num_creases: AnyDist
    strength: AnyDist
    angle_deg: AnyDist
    position_factor: AnyDist
    crease_width_factor: AnyDist # <-- ADD THIS LINE
    crease_center_x_factor: AnyDist # Horizontal center of the crease
    crease_length_factor: AnyDist # Length of the crease
# --- END NEW ---

class TextBoxDistortionConfig(BaseModel):
    shear: ShearConfig
    stretch: StretchConfig
    warp_curl: WarpCurlConfig
    # --- NEW: Add the new config to the main distortion model ---
    linear_crease: LinearCreaseConfig
    # --- END NEW ---

class PageAugmentationsConfig(BaseModel):
    orientation_deg: AnyDist
    orientation_other_range: AnyDist
    point_dropout: Dict[str, Any]


class SplitAugmentationConfig(BaseModel):
    """Configuration for augmenting one of the two split textboxes to create a signal."""
    enabled: bool = True
    font_size_factor: AnyDist = Field(default_factory=lambda: {"dist": "uniform_float", "min": 0.8, "max": 1.2})
    
    # --- NEW: Scaling and Stretching Factors ---
    stretch_x_factor: AnyDist = Field(default_factory=lambda: {"dist": "uniform_float", "min": 0.8, "max": 1.2})
    stretch_y_factor: AnyDist = Field(default_factory=lambda: {"dist": "uniform_float", "min": 0.8, "max": 1.2})
    uniform_scale_factor: AnyDist = Field(default_factory=lambda: {"dist": "uniform_float", "min": 0.9, "max": 1.1})

    horizontal_split_shift_x_factor: AnyDist = Field(default_factory=lambda: {"dist": "uniform_float", "min": -0.3, "max": 0.3})
    horizontal_split_shift_y_factor: AnyDist = Field(default_factory=lambda: {"dist": "uniform_float", "min": -0.1, "max": 0.1})
    vertical_split_shift_x_factor: AnyDist = Field(default_factory=lambda: {"dist": "uniform_float", "min": -0.1, "max": 0.1})
    vertical_split_shift_y_factor: AnyDist = Field(default_factory=lambda: {"dist": "uniform_float", "min": -0.3, "max": 0.3})
    
    internal_crop_scale_factor: AnyDist = Field(default_factory=lambda: {"dist": "uniform_float", "min": 0.85, "max": 0.98})

    rotation_deg: AnyDist = Field(default_factory=lambda: {"dist": "uniform_float", "min": -3.0, "max": 3.0})

    line_reduction_ratio: AnyDist = Field(default_factory=lambda: {"dist": "uniform_float", "min": 0.0, "max": 0.3})

    line_cut_short_prob: AnyDist = Field(default_factory=lambda: {"dist": "uniform_float", "min": 0.1, "max": 0.5})
    line_cut_short_factor: AnyDist = Field(default_factory=lambda: {"dist": "uniform_float", "min": 0.05, "max": 0.25})

class TextBoxSplittingConfig(BaseModel):
    """Main configuration for the textbox splitting feature."""
    enabled: bool = False
    probability: float = Field(0.5, description="Probability to attempt a split on a given textbox.")
    
    # --- NEW: Bias for aspect ratio based splitting ---
    aspect_ratio_bias: float = Field(0.4, description="Value from 0 to 0.5. Controls how strongly aspect ratio influences split direction. Higher value means stronger bias.")

    split_method_probabilities: Dict[str, float] = Field(default_factory=lambda: {
        "horizontal": 0.45,
        "vertical": 0.45,
        "internal_crop": 0.1
    })
    augmentation: SplitAugmentationConfig = Field(default_factory=SplitAugmentationConfig)

class Config(BaseModel):
    generation: GenerationConfig
    visualization: VisualizationConfig
    page: PageConfig
    rejection_sampling: RejectionSamplingConfig
    grid_layout: GridLayoutConfig
    textbox_content: TextBoxContentConfig
    textbox_distortion: TextBoxDistortionConfig
    page_augmentations: PageAugmentationsConfig
    # --- NEW: Add the splitting config to the main Config model ---
    textbox_splitting: TextBoxSplittingConfig
    # --- END NEW ---