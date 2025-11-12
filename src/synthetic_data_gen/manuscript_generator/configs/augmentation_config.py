# augmentation_config.py

import sys
from pathlib import Path
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Union, Literal

# To reuse the type definitions from the existing codebase without modifying it,
# we add the project root to the path. This assumes the script is run from the project root.
sys.path.append(str(Path(__file__).parent.resolve()))

# --- Re-defining necessary Pydantic Models for type validation ---
# This avoids modifying existing files and makes this config self-contained.
# These are copies of the models in manuscript_generator/configs/base_config.py
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

# --- Augmentation-Specific Configuration Models ---

class GeneralConfig(BaseModel):
    """General settings for the augmentation script."""
    input_dir: str = "./data/real-dataset"
    output_dir: str = "./data/augmented-dataset"
    
    # --- MODIFIED: Added the required validation output directory field ---
    val_output_dir: str = "./data/validation-dataset"
    
    num_augmentations_per_sample: int = 5
    num_workers: int = -1
    base_seed: int = 42
    visualize: bool = False
    visualize_every_n: int = 10

class AugmentationParam(BaseModel):
    """Base model for an augmentation, ensuring it has enable and probability flags."""
    enabled: bool
    probability: float

    @validator('probability')
    def probability_bounds(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('probability must be between 0.0 and 1.0')
        return v

# --- Phase 1: Content Augmentations ---
class FontSizeVariationConfig(AugmentationParam):
    variation_factor: AnyDist = Field(..., description="Multiplicative factor for font size noise.")

class PointLevelJitterConfig(AugmentationParam):
    jitter_std_factor: AnyDist = Field(..., description="Jitter std dev, relative to font size.")

class Phase1Config(BaseModel):
    """Configuration for point-level content augmentations."""
    font_size_variation: FontSizeVariationConfig
    point_level_jitter: PointLevelJitterConfig

# --- Phase 2: Geometric Distortion (Applied at Page Level) ---
# We reuse the detailed config models from the generator.
class ShearConfig(AugmentationParam):
    shear_factor_x: AnyDist
    shear_factor_y: AnyDist

class StretchConfig(AugmentationParam):
    stretch_factor_x: AnyDist
    stretch_factor_y: AnyDist

class WarpCurlConfig(AugmentationParam):
    amplitude_factor: AnyDist
    frequency_factor_x: AnyDist
    frequency_factor_y: AnyDist
    phase_x: AnyDist
    phase_y: AnyDist

class LinearCreaseConfig(AugmentationParam):
    num_creases: AnyDist
    strength: AnyDist
    angle_deg: AnyDist
    position_factor: AnyDist
    crease_width_factor: AnyDist
    crease_center_x_factor: AnyDist
    crease_length_factor: AnyDist

class Phase2Config(BaseModel):
    """Configuration for page-level geometric distortions."""
    shear: ShearConfig
    stretch: StretchConfig
    warp_curl: WarpCurlConfig
    linear_crease: LinearCreaseConfig

# --- Phase 3: Page-level Dropout ---
class PointDropoutConfig(AugmentationParam):
    # CORRECTED: Removed the confusing alias. Now the field name is distinct.
    dropout_probability_dist: AnyDist = Field(..., description="Distribution to sample the dropout probability from.")

class Phase3Config(BaseModel):
    """Configuration for page-level structural augmentations like dropout."""
    point_dropout: PointDropoutConfig

# --- Phase 4: New Page-level Transforms ---
class PageRotationConfig(AugmentationParam):
    rotation_deg: AnyDist = Field(..., description="Rotation angle in degrees.")

class PageTranslationConfig(AugmentationParam):
    translate_x_factor: AnyDist = Field(..., description="Translation in X, as a factor of page width.")
    translate_y_factor: AnyDist = Field(..., description="Translation in Y, as a factor of page height.")

class PageMirrorConfig(AugmentationParam):
    horizontal_prob: float = 0.5
    vertical_prob: float = 0.5

class Phase4Config(BaseModel):
    """Configuration for new, real-data-specific page-level transforms."""
    page_rotation: PageRotationConfig
    page_translation: PageTranslationConfig
    page_mirror: PageMirrorConfig

# --- Main Configuration Class ---
class AugmentationConfig(BaseModel):
    """Root model for the entire augmentation configuration."""
    general: GeneralConfig
    phase1_content: Phase1Config
    phase2_distortion: Phase2Config
    phase3_page: Phase3Config
    phase4_page: Phase4Config