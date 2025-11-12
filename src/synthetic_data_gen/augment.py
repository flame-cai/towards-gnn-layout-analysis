# augment.py

import os
import sys
import yaml
import time
import logging
import argparse
import numpy as np
import shutil
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count
from types import SimpleNamespace

from tqdm import tqdm

# ... (rest of your imports remain the same) ...
from manuscript_generator.augmentations import phase1_content, phase2_distortion, phase3_page
from manuscript_generator.augmentations import phase4_page
from manuscript_generator.core.registry import AUGMENTATIONS
from manuscript_generator.core.textbox import TextBox, TextBoxType
from manuscript_generator.utils.plotter import visualize_page, Page
from manuscript_generator.configs.augmentation_config import AugmentationConfig


def setup_logging():
    """Configures logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_real_page_data(page_id: str, base_path: Path) -> (dict, np.ndarray, np.ndarray):
    """Loads all data files for a single real-data page."""
    try:
        dims_path = base_path / f"{page_id}_dims.txt"
        inputs_path = base_path / f"{page_id}_inputs_unnormalized.txt"
        labels_path = base_path / f"{page_id}_labels_textline.txt"

        assert dims_path.exists(), f"Dimension file not found: {dims_path}"
        assert inputs_path.exists(), f"Inputs file not found: {inputs_path}"
        assert labels_path.exists(), f"Labels file not found: {labels_path}"

        dims_arr = np.loadtxt(dims_path)

        # Check if we actually got at least two numbers
        if dims_arr.size < 2:
            raise ValueError(f"Dimensions file '{dims_path}' is malformed. Expected 2 values, got {dims_arr.size}.")

        # Now it is safe to proceed
        dims = {"width": dims_arr[0], "height": dims_arr[1]}
        
        points = np.loadtxt(inputs_path, ndmin=2)
        labels = np.loadtxt(labels_path, dtype=int, ndmin=1)

        assert points.shape[0] == labels.shape[0], \
            f"Page {page_id}: Mismatch between points ({points.shape[0]}) and labels ({labels.shape[0]})"
        assert points.shape[1] == 3, f"Page {page_id}: Points data must have 3 columns (x, y, s)"

        return dims, points, labels
    except Exception as e:
        logging.error(f"Failed to load data for page {page_id}: {e}")
        return None, None, None

def save_augmented_data(page_id: str, output_dir: Path, dims: dict, points: np.ndarray, labels: np.ndarray):
    """Saves the augmented data for a single page to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    longest_dim = max(dims['width'], dims['height'])
    assert longest_dim > 0, "Page dimensions must be positive."

    points_normalized = points.copy()
    points_normalized[:, :2] /= longest_dim
    points_normalized[:, 2] = points[:, 2] 

    np.savetxt(output_dir / f"{page_id}_inputs_unnormalized.txt", points, fmt="%.2f %.2f %d")
    np.savetxt(output_dir / f"{page_id}_inputs_normalized.txt", points_normalized, fmt="%.6f %.6f %.6f")
    np.savetxt(output_dir / f"{page_id}_labels_textline.txt", labels, fmt="%d")
    with open(output_dir / f"{page_id}_dims.txt", 'w') as f:
        f.write(f"{dims['width']} {dims['height']}")

def copy_original_data(page_id: str, input_dir: Path, output_dir: Path):
    """Copies the three original data files for a page to a new directory."""
    try:
        file_suffixes = ["_dims.txt", "_inputs_unnormalized.txt","_inputs_normalized.txt", "_labels_textline.txt"]
        for suffix in file_suffixes:
            source_file = input_dir / f"{page_id}{suffix}"
            if source_file.exists():
                shutil.copy(source_file, output_dir)
            else:
                logging.warning(f"Source file not found for copying: {source_file}")
    except Exception as e:
        logging.error(f"Failed to copy data for page {page_id}: {e}")

def _augment_single_instance(task_info: tuple):
    """
    Worker function to perform augmentation on a single real data sample.
    """
    page_id, page_index, aug_idx, seed, config, input_dir, output_dir = task_info
    rng = np.random.default_rng(seed)
    
    dims, points, labels = load_real_page_data(page_id, input_dir)
    if dims is None:
        return f"Skipped page {page_id} due to loading error."

    aug_points, aug_labels = points.copy(), labels.copy()

    # --- AUGMENTATION PIPELINE (UNCHANGED) ---
    config_adapter = SimpleNamespace()
    config_adapter.textbox_content = SimpleNamespace(
        font_size_variation=config.phase1_content.font_size_variation.model_dump(),
        point_level_jitter=config.phase1_content.point_level_jitter.model_dump(),
        congestion_jitter={'enabled': False}
    )
    config_adapter.textbox_distortion = config.phase2_distortion
    p3_dropout_config = config.phase3_page.point_dropout
    config_adapter.page_augmentations = SimpleNamespace(
        point_dropout={
            'enabled': p3_dropout_config.enabled,
            'probability': p3_dropout_config.dropout_probability_dist.model_dump()
        }
    )
    p1_config = config.phase1_content
    if p1_config.font_size_variation.enabled and rng.random() < p1_config.font_size_variation.probability:
        aug_points = AUGMENTATIONS['font_size_variation'](aug_points, aug_labels, config_adapter, rng)
    if p1_config.point_level_jitter.enabled and rng.random() < p1_config.point_level_jitter.probability:
        aug_points = AUGMENTATIONS['point_level_jitter'](aug_points, aug_labels, config_adapter, rng)
    p2_config = config.phase2_distortion
    distortions = list(p2_config.model_dump().keys())
    rng.shuffle(distortions)
    center_transform = np.array([dims['width'] / 2.0, dims['height'] / 2.0, 0])
    centered_points = aug_points - center_transform
    dummy_textbox = TextBox(box_type=TextBoxType.MAIN_TEXT)
    dummy_textbox.points_local = centered_points
    dummy_textbox.width = dims['width']
    dummy_textbox.height = dims['height']
    for aug_name in distortions:
        aug_conf = getattr(p2_config, aug_name)
        if aug_conf.enabled and rng.random() < aug_conf.probability:
            dummy_textbox.points_local = AUGMENTATIONS[aug_name](dummy_textbox.points_local, dummy_textbox, config_adapter, rng)
    aug_points = dummy_textbox.points_local + center_transform
    p4_config = config.phase4_page
    page_dims_dict = {'width': dims['width'], 'height': dims['height']}
    for aug_name, aug_conf in [
        ("page_rotation", p4_config.page_rotation),
        ("page_translation", p4_config.page_translation),
        ("page_mirror", p4_config.page_mirror)
    ]:
         if aug_conf.enabled and rng.random() < aug_conf.probability:
            aug_points = AUGMENTATIONS[aug_name](aug_points, page_dims_dict, config, rng)
    p3_config = config.phase3_page.point_dropout
    if p3_config.enabled and rng.random() < p3_config.probability:
        aug_points, kept_indices = AUGMENTATIONS['point_dropout'](aug_points, None, config_adapter, rng)
        aug_labels = aug_labels[kept_indices]
    # --- END AUGMENTATION PIPELINE ---

    aug_page_id = f"{page_id}_{aug_idx}"
    save_augmented_data(aug_page_id, output_dir, dims, aug_points, aug_labels)
    
    if config.general.visualize and (page_index * config.general.num_augmentations_per_sample + aug_idx) % config.general.visualize_every_n == 0:
        viz_page = Page(width=int(dims['width']), height=int(dims['height']))
        viz_page.points = aug_points
        viz_page.textline_labels = aug_labels
        
        class DummyVizConfig:
            coloring = "textline"
            point_size_multiplier = 1.0
        
        viz_path = output_dir / f"{aug_page_id}.png"
        visualize_page(viz_page, DummyVizConfig(), viz_path)

    return None

def main():
    """Main execution function to run the augmentation pipeline."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Split, augment, and save manuscript data.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/augmentation.yaml',
        help='Path to the YAML configuration file for augmentations.'
    )
    # --- NEW: Add command-line arguments for directories ---
    parser.add_argument('--input_dir', type=str, help='Override the input directory from the config file.')
    parser.add_argument('--output_dir', type=str, help='Override the output directory for training data from the config file.')
    parser.add_argument('--val_output_dir', type=str, help='Override the output directory for validation data from the config file.')

    args = parser.parse_args()

    config_path = Path(args.config)
    assert config_path.exists(), f"Configuration file not found at {config_path}"

    logging.info(f"Loading augmentation configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    try:
        config = AugmentationConfig(**config_dict)
    except Exception as e:
        logging.critical(f"Error validating configuration: {e}")
        return

    # --- MODIFIED: Prioritize command-line arguments over config values ---
    input_dir = Path(args.input_dir) if args.input_dir else Path(config.general.input_dir)
    train_output_dir = Path(args.output_dir) if args.output_dir else Path(config.general.output_dir)
    
    try:
        val_output_dir_config = config.general.val_output_dir
    except AttributeError:
        logging.critical("Configuration error: `general.val_output_dir` is not defined in the YAML file.")
        return
        
    val_output_dir = Path(args.val_output_dir) if args.val_output_dir else Path(val_output_dir_config)

    train_output_dir.mkdir(parents=True, exist_ok=True)
    # val_output_dir.mkdir(parents=True, exist_ok=True)

    assert input_dir.is_dir(), f"Input directory not found: {input_dir}"

    files = os.listdir(input_dir)
    page_ids = sorted(list(set([f.removesuffix('_dims.txt') for f in files if f.endswith('_dims.txt')])))

    logging.info(f"Found {len(page_ids)} total pages in '{input_dir}'.")
    if not page_ids:
        logging.warning("No pages found. Exiting.")
        return

    # --- NEW: Shuffle and split the data ---
    logging.info("Shuffling and splitting dataset into 0.8 train and 0.2 validation...")
    rng = np.random.default_rng(config.general.base_seed)
    shuffled_page_ids = np.array(page_ids, dtype=object)
    rng.shuffle(shuffled_page_ids)

    split_index = int(len(shuffled_page_ids) * 1.0)
    train_page_ids = shuffled_page_ids[:split_index]
    val_page_ids = shuffled_page_ids[split_index:]

    logging.info(f"Training set size: {len(train_page_ids)} pages.")
    logging.info(f"Validation set size: {len(val_page_ids)} pages.")

    # --- NEW: Process and save the validation set (copy only) ---
    # logging.info(f"Copying {len(val_page_ids)} validation samples to '{val_output_dir}'...")
    # for page_id in tqdm(val_page_ids, desc="Copying validation data"):
    #     copy_original_data(page_id, input_dir, val_output_dir)

    # --- MODIFIED: Process the training set (copy original + augment) ---
    logging.info(f"Processing {len(train_page_ids)} training samples for augmentation...")

    # First, copy all original training files
    for page_id in tqdm(train_page_ids, desc="Copying original training data"):
        copy_original_data(page_id, input_dir, train_output_dir)

    # Second, create augmentation tasks
    tasks = []
    for page_index, page_id in enumerate(train_page_ids):
        for i in range(config.general.num_augmentations_per_sample):
            seed = config.general.base_seed + page_index * config.general.num_augmentations_per_sample + i
            tasks.append((page_id, page_index, i, seed, config, input_dir, train_output_dir))

    logging.info(f"Total augmentations to generate: {len(tasks)}")

    start_time = time.time()
    num_workers = config.general.num_workers
    if num_workers == -1:
        num_workers = cpu_count()

    if num_workers > 1 and len(tasks) > 1:
        logging.info(f"Using {num_workers} parallel workers for augmentation.")
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(_augment_single_instance, tasks), total=len(tasks), desc="Augmenting Data"))
    else:
        logging.info("Using a single process for augmentation (sequential).")
        results = [_augment_single_instance(task) for task in tqdm(tasks, desc="Augmenting Data")]

    duration = time.time() - start_time
    errors = [r for r in results if r is not None]

    logging.info(f"\n--- Augmentation Complete ---")
    logging.info(f"Total time for augmentation: {duration:.2f} seconds.")
    logging.info(f"Successfully generated: {len(tasks) - len(errors)} augmented samples.")
    if errors:
        logging.warning(f"Encountered {len(errors)} errors during augmentation. See logs for details.")

if __name__ == "__main__":
    main()