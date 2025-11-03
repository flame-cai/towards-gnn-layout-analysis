import os
import re
import logging
import argparse
from pathlib import Path
from lxml import etree as ET
from shapely.geometry import Polygon, Point, MultiPolygon, LineString
from itertools import combinations
from PIL import Image, ImageDraw

# --- Start of User-Provided Polygon Processing Functions ---

def _convert_to_thinner_rectangle(polygon: Polygon) -> list[Polygon]:
    """
    Converts a single polygon to its minimum area rotated bounding rectangle,
    then reduces its thickness by half while maintaining the same center.
    This is a helper function.

    :param polygon: The initial Shapely polygon.
    :return: A list containing the single refined rectangle polygon, or an empty list on failure.
    """
    try:
        if not polygon.is_valid or polygon.is_empty:
            return []

        # Step 1: Get the minimum rotated rectangle.
        rect_poly = polygon.minimum_rotated_rectangle
        if not rect_poly.is_valid or rect_poly.is_empty:
            return []

        # Step 2: Get the four corner points of the rectangle.
        corners = list(rect_poly.exterior.coords)[:4]
        p = [Point(c) for c in corners]

        # Step 3: Identify the long and short sides to determine "thickness".
        dist_sq_1 = p[0].distance(p[1]) ** 2
        dist_sq_2 = p[1].distance(p[2]) ** 2

        if dist_sq_1 > dist_sq_2:
            p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
        else:
            p0, p1, p2, p3 = p[1], p[2], p[3], p[0]

        # Step 4: Get the center and define the half-axis vectors.
        center = rect_poly.centroid
        midpoint_long_side = Point((p0.x + p1.x) / 2, (p0.y + p1.y) / 2)
        v_half_short_axis = (midpoint_long_side.x - center.x, midpoint_long_side.y - center.y)
        midpoint_short_side = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
        v_half_long_axis = (midpoint_short_side.x - center.x, midpoint_short_side.y - center.y)

        # Step 5: Scale the short axis vector by 0.5 to halve the thickness.
        v_half_short_axis_scaled = (v_half_short_axis[0] * 0.5, v_half_short_axis[1] * 0.5)

        # Step 6: Calculate the new corner points around the original center.
        c = (center.x, center.y)
        v_long = v_half_long_axis
        v_short = v_half_short_axis_scaled
        new_corners = [
            (c[0] - v_long[0] + v_short[0], c[1] - v_long[1] + v_short[1]),
            (c[0] + v_long[0] + v_short[0], c[1] + v_long[1] + v_short[1]),
            (c[0] + v_long[0] - v_short[0], c[1] + v_long[1] - v_short[1]),
            (c[0] - v_long[0] - v_short[0], c[1] - v_long[1] - v_short[1]),
        ]

        # Step 7: Create the new, thinner polygon.
        thinner_rect = Polygon(new_corners)
        if thinner_rect.is_valid and not thinner_rect.is_empty:
            return [thinner_rect]
        else:
            logging.warning("Could not generate a valid thinner rectangle, returning original rotated rect.")
            return [rect_poly]
    except Exception as e:
        logging.error(f"Failed during rectangle conversion: {e}")
        return []

def split_polygons(polygons: list) -> list:
    """
    First, converts all input polygons to thinner, rotated rectangles.
    Then, splits any touching and overlapping rectangles using an erosion-based method.

    :param polygons: The list of original shapely polygons to process.
    :return: A list of non-touching, processed rectangle polygons.
    """
    # --- STAGE 1: Convert all polygons to thinner rectangles ---
    rectangles = []
    for poly in polygons:
        rectangles.extend(_convert_to_thinner_rectangle(poly))
    
    if not rectangles:
        return []

    # --- STAGE 2: Apply the original erosion and splitting logic to the rectangles ---
    eps = 2
    processed_rects = list(rectangles)

    for comb in combinations(range(len(processed_rects)), 2):
        poly1 = processed_rects[comb[0]]
        poly2 = processed_rects[comb[1]]

        if not poly1.is_valid or not poly2.is_valid:
            continue
        
        if poly1.intersects(poly2):
            buffered_poly1 = poly1.buffer(-eps)
            buffered_poly2 = poly2.buffer(-eps)
            intersection = buffered_poly1.intersection(buffered_poly2)
            
            if not intersection.is_empty:
                if (isinstance(intersection, Polygon) and
                    intersection.area < 0.2 * buffered_poly1.area and
                    intersection.area < 0.2 * buffered_poly2.area) or \
                   isinstance(intersection, MultiPolygon):
                    
                    if buffered_poly1.area > buffered_poly2.area:
                        processed_rects[comb[0]] = buffered_poly1.difference(intersection)
                        processed_rects[comb[1]] = buffered_poly2
                    else:
                        processed_rects[comb[1]] = buffered_poly2.difference(intersection)
                        processed_rects[comb[0]] = buffered_poly1
                elif isinstance(intersection, LineString):
                    processed_rects[comb[0]] = buffered_poly1.difference(intersection)
                    processed_rects[comb[1]] = buffered_poly2.difference(intersection)

        elif poly1.touches(poly2):
            processed_rects[comb[0]] = poly1.buffer(-2 * eps)
            processed_rects[comb[1]] = poly2.buffer(-2 * eps)

    final_rects = [poly.buffer(-2 * eps) for poly in processed_rects]
    
    return [p for p in final_rects if p.is_valid and not p.is_empty]


def apply_split_rects_processing(polygons_with_conf: list[tuple[float, Polygon]]) -> list[tuple[float, Polygon]]:
    if not polygons_with_conf:
        return []
    logging.info("Applying split-and-thin rectangle processing. Note: Original confidence scores will be reset to 1.0.")
    polys_only = [poly for _, poly in polygons_with_conf]
    processed_polys = split_polygons(polys_only)
    return [(1.0, poly) for poly in processed_polys]

# --- End of User-Provided Functions ---


def parse_xml_polygons(xml_path: Path, ns: dict) -> list[tuple[float, Polygon]]:
    """
    Parses a PAGE XML file to extract all TextLine polygons.
    """
    polygons = []
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        for textline in root.findall(".//page:TextLine", ns):
            coords = textline.find("./page:Coords", ns)
            if coords is not None and "points" in coords.attrib:
                confidence = float(coords.attrib.get("conf", 1.0))
                points_str = coords.attrib["points"].split()
                points = [(int(p.split(",")[0]), int(p.split(",")[1])) for p in points_str]
                if len(points) >= 3:
                    poly = Polygon(points).buffer(0)
                    if not poly.is_empty:
                        polygons.append((confidence, poly))
    except Exception as e:
        logging.error(f"Failed to parse XML {xml_path}: {e}")
    return polygons


def save_processed_xml(original_xml_path: Path, output_xml_path: Path, processed_geometries: list, ns: dict):
    """
    Saves the processed polygons into a new PAGE XML file,
    preserving the structure of the original and correctly handling MultiPolygons.
    """
    try:
        tree = ET.parse(str(original_xml_path))
        root = tree.getroot()
        
        # Clear existing TextLine elements to replace them with new ones
        for page in root.findall(".//page:Page", ns):
            for text_region in page.findall(".//page:TextRegion", ns):
                # Using a list comprehension to avoid issues with modifying while iterating
                for text_line in text_region.findall(".//page:TextLine", ns):
                    text_region.remove(text_line)

        # Find the first TextRegion to add the new lines to.
        # This might need adjustment for more complex XML structures.
        text_region = root.find(".//page:TextRegion", ns)
        if text_region is None:
            page = root.find(".//page:Page", ns)
            if page is None:
                logging.error(f"No <Page> element found in {original_xml_path}. Cannot save processed lines.")
                return
            # If no TextRegion exists, create one to hold the new lines.
            text_region = ET.SubElement(page, "TextRegion", {"id": "processed_region_1"})
            ET.SubElement(text_region, "Coords", {"points": ""}) # Add empty Coords for schema validity

        # --- SOLUTION ---
        # Flatten the list: convert any MultiPolygon into a series of Polygons.
        # This elegantly handles the error case.
        flat_polygons = []
        for geom in processed_geometries:
            if isinstance(geom, Polygon):
                flat_polygons.append(geom)
            elif isinstance(geom, MultiPolygon):
                # If it's a MultiPolygon, add each of its constituent polygons to the list.
                flat_polygons.extend(list(geom.geoms))
        
        # Now, iterate over the flattened list of guaranteed Polygon objects.
        for i, poly in enumerate(flat_polygons):
            # Ensure the polygon is valid and has an exterior before proceeding
            if poly.is_empty or poly.exterior is None:
                continue

            points_str = " ".join([f"{int(x)},{int(y)}" for x, y in poly.exterior.coords])
            
            # Create a new TextLine for each processed polygon
            textline_element = ET.SubElement(text_region, "TextLine", {"id": f"line_processed_{i+1}"})
            coords_element = ET.SubElement(textline_element, "Coords", {"points": points_str})

        tree.write(str(output_xml_path), pretty_print=True, xml_declaration=True, encoding="UTF-8")

    except Exception as e:
        # The specific AttributeError should now be resolved, but this general catch remains for other issues.
        logging.error(f"Failed to write processed XML {output_xml_path}: {e}")

def visualize_polygons(image_path: Path, output_path: Path, original_polygons: list, processed_polygons: list):
    """
    Draws original and processed polygons on the source image and saves it.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for _, poly in original_polygons:
             if poly.exterior:
                coords = list(poly.exterior.coords)
                draw.polygon(coords, outline="red", width=2)
        
        for _, poly in processed_polygons:
            if poly.exterior:
                coords = list(poly.exterior.coords)
                draw.polygon(coords, outline="lime", width=2)

        img.save(output_path)
        logging.info(f"Saved visualization to {output_path}")

    except FileNotFoundError:
        logging.warning(f"Image file not found: {image_path}. Skipping visualization.")
    except Exception as e:
        logging.error(f"Failed to create visualization for {image_path}: {e}")

def main():
    """
    Main function to orchestrate the parsing, processing, and saving of XML files.
    """
    parser = argparse.ArgumentParser(description="Process PAGE XML polygon data from a directory.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the input PAGE XML files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the processed XML files will be saved.")
    parser.add_argument("--image_dir", type=str, help="Optional: Directory with original images for visualization.")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    image_path = Path(args.image_dir) if args.image_dir else None

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Validate paths
    assert input_path.is_dir(), f"Input directory not found: {input_path}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    if image_path:
        assert image_path.is_dir(), f"Image directory not found: {image_path}"
        visualization_path = output_path.parent / "visualization_check"
        visualization_path.mkdir(exist_ok=True)
        logging.info(f"Visualizations will be saved in: {visualization_path}")

    # Discover XML files
    xml_files = sorted(list(input_path.glob("*.xml")))
    assert len(xml_files) > 0, "No XML files found in the input directory."
    
    # Define the PAGE XML namespace
    ns = {"page": "https://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
    
    processed_count = 0
    for xml_file in xml_files:
        logging.info(f"Processing {xml_file.name}...")
        
        # 1. Parse original polygons
        original_polygons_with_conf = parse_xml_polygons(xml_file, ns)
        if not original_polygons_with_conf:
            logging.warning(f"No polygons found in {xml_file.name}. Skipping.")
            continue
            
        # 2. Process polygons
        processed_polygons_with_conf = apply_split_rects_processing(original_polygons_with_conf)
        processed_polygons = [poly for _, poly in processed_polygons_with_conf]
        
        # 3. Save processed XML
        output_xml_file = output_path / xml_file.name
        save_processed_xml(xml_file, output_xml_file, processed_polygons, ns)
        
        # 4. Optional: Visualize results
        if image_path:
            # Assumes image has the same basename but a common extension (.jpg, .png, etc.)
            image_file = next(image_path.glob(f"{xml_file.stem}.*"), None)
            if image_file:
                vis_output_file = visualization_path / f"{xml_file.stem}_visualization.jpg"
                visualize_polygons(image_file, vis_output_file, original_polygons_with_conf, processed_polygons_with_conf)
            else:
                logging.warning(f"No corresponding image found for {xml_file.name} in {image_path}")

        processed_count += 1

    logging.info("Processing complete.")
    assert len(list(output_path.glob("*.xml"))) == processed_count, "Mismatch in number of input and output XML files."
    logging.info(f"Successfully processed {processed_count} files.")


if __name__ == "__main__":
    main()