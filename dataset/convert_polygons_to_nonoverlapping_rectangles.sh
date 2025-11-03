#!/bin/bash

# --- Configuration ---
# Add all your manuscript names to this list.
# Ensure each name corresponds to a directory at the same level as the script.
MANUSCRIPTS=(
    "amarakoshah-kanda2-3"
    "amaranathamahatmyam"
    "ashirmanimala"
    "bhagavatamahatmyam-padmapuranagatam1"
    "bhagavatapuranam-sateekam-skandha-1-1mashlokah"
    "dashabhuvanapatayah"
    "pakshikachaturmasika-samvatsarika-pratikramanavidhah"
    "ravisankrantivicharah"
    "tarkasangrahah-sateekah2"
    "vagbhatalamkara-vyakhya1"
    "vrittaratnakarah"
    "vrittaratnakarah1"
    "vyapati-sang-rahah-muktavaligatavyapati-vyakhya"
    "yajnavalkyashiksha"
    "yujurvedashakavrikshah-chitram"
)


# The name of the Python script to execute.
PYTHON_SCRIPT="convert_polygons_to_nonoverlapping_rectangles.py"

# --- Behavioral Flags ---
# Set to 'true' to automatically delete all *.xml files in the output directory before running.
CLEAN_OUTPUT_BEFORE_RUN=true

# --- Script Logic ---

# For colorful output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Arrays to hold the names of processed manuscripts for the final summary.
SUCCESS_LIST=()
FAIL_LIST=()

# --- Pre-flight Check ---
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: The script '$PYTHON_SCRIPT' was not found in the current directory.${NC}"
    exit 1
fi

echo "Starting batch processing for ${#MANUSCRIPTS[@]} manuscripts..."
echo "--------------------------------------------------------"

# --- Main Processing Loop ---
for manuscript_name in "${MANUSCRIPTS[@]}"; do
    
    echo "Processing manuscript: $manuscript_name"

    # Define the directory paths
    INPUT_DIR="${manuscript_name}/page-xml-graph-groundtruth"
    OUTPUT_DIR="${manuscript_name}/page-xml-rectangle"
    IMAGE_DIR="${manuscript_name}/images"

    # --- Pre-run Directory Checks ---
    if [ ! -d "$manuscript_name" ] || [ ! -d "$INPUT_DIR" ]; then
        echo -e "  ${RED}[FAILURE]${NC} Main directory or input directory not found. Skipping."
        FAIL_LIST+=("$manuscript_name (Setup Error)")
        echo "--------------------------------------------------------"
        continue
    fi

    # --- Clean Output Directory (Optional) ---
    if [ "$CLEAN_OUTPUT_BEFORE_RUN" = true ] && [ -d "$OUTPUT_DIR" ]; then
        echo "  Cleaning output directory..."
        # Safely remove only xml files from the directory
        rm -f "${OUTPUT_DIR}"/*.xml
    fi
    # Ensure the output directory exists
    mkdir -p "$OUTPUT_DIR"

    # --- Count Input Files for Validation ---
    # Use find for robustness, handles cases with no files gracefully
    num_input_files=$(find "$INPUT_DIR" -maxdepth 1 -name "*.xml" | wc -l)
    if [ "$num_input_files" -eq 0 ]; then
        echo -e "  ${RED}[FAILURE]${NC} No XML files found in the input directory. Skipping."
        FAIL_LIST+=("$manuscript_name (No Input Files)")
        echo "--------------------------------------------------------"
        continue
    fi
    echo "  Found $num_input_files XML files to process."

    # --- Handle Optional Image Directory ---
    IMAGE_ARG=""
    if [ -d "$IMAGE_DIR" ]; then
        IMAGE_ARG="--image_dir ${IMAGE_DIR}"
        echo "  Found image directory, visualization will be enabled."
    else
        echo "  Image directory not found, proceeding without visualization."
    fi

    # --- Execute the Python Script ---
    # 'eval' is used to correctly handle IMAGE_ARG which might be empty
    eval python "$PYTHON_SCRIPT" \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR"
        # $IMAGE_ARG
    
    # Capture the exit code of the last command
    exit_code=$?

    # --- Validate the Outcome ---
    num_output_files=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.xml" | wc -l)

    if [ $exit_code -eq 0 ] && [ "$num_input_files" -eq "$num_output_files" ]; then
        echo -e "  ${GREEN}[SUCCESS]${NC} Finished processing $manuscript_name."
        SUCCESS_LIST+=("$manuscript_name")
    else
        echo -e "  ${RED}[FAILURE]${NC} Processing failed for $manuscript_name."
        echo -e "    - Python script exit code: $exit_code"
        echo -e "    - Input files: $num_input_files, Output files: $num_output_files"
        FAIL_LIST+=("$manuscript_name")
    fi
    echo "--------------------------------------------------------"
done

# --- Final Summary ---
echo ""
echo "========================================================"
echo "                 BATCH PROCESSING SUMMARY"
echo "========================================================"
echo -e "Successfully processed: ${GREEN}${#SUCCESS_LIST[@]}${NC} manuscript(s)."
echo -e "Failed to process:    ${RED}${#FAIL_LIST[@]}${NC} manuscript(s)."

if [ ${#FAIL_LIST[@]} -ne 0 ]; then
    echo ""
    echo -e "${RED}The following manuscripts failed:${NC}"
    for failed_manuscript in "${FAIL_LIST[@]}"; do
        echo "  - $failed_manuscript"
    done
    echo "Please review the logs above for specific errors."
    exit 1 # Exit with a failure code
fi

echo -e "\n${GREEN}All manuscripts processed successfully!${NC}"
exit 0