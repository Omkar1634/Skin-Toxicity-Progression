import cv2
import numpy as np

def create_image_grid(image_paths: list[str], cols: int = 3, tile_size: tuple[int, int] = (300, 300), output_path: str = "grid_result.jpg"):
    """
    Combines a list of image paths into a single grid image.

    Parameters:
        image_paths (list): List of paths to the images.
        cols (int): Number of columns in the grid.
        tile_size (tuple): (width, height) to resize each individual image to.
        output_path (str): File path to save the merged grid.
    """
    images = []
    
    # Load and resize each image
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not load image at '{path}', skipping.")
            continue
        resized_img = cv2.resize(img, tile_size, interpolation=cv2.INTER_AREA)
        images.append(resized_img)

    if not images:
        raise ValueError("No valid images were loaded.")

    # Calculate rows needed
    num_images = len(images)
    rows = (num_images + cols - 1) // cols  # Ceiling division

    # Fill empty slots with blank (black) tiles if images don't fill the final row
    blank_tile = np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
    while len(images) < rows * cols:
        images.append(blank_tile)

    # Build the grid row by row
    row_images = []
    for i in range(rows):
        row = np.hstack(images[i * cols : (i + 1) * cols])
        row_images.append(row)

    # Stack all rows vertically
    grid = np.vstack(row_images)

    # Save and return the output image
    cv2.imwrite(output_path, grid)
    print(f"Grid saved successfully to '{output_path}'.")
    return grid


# Example usage with 5 images arranged in a 3-column layout (2 rows: 3 in row 1, 2 + blank in row 2):
if __name__ == "__main__":
    paths = [
        "D:\\Github\\PhD Code\\Erythema-Progression\\output\\different_skin\\oo\\progression\\2026-09-09_13-31-51\\frame_08_amp0.15.jpeg",
        "D:\\Github\\PhD Code\\Erythema-Progression\\output\\different_skin\\22\\progression\\2026-09-09_13-33-18\\frame_08_amp0.15.jpeg",
        "D:\\Github\\PhD Code\\Erythema-Progression\\output\\different_skin\\37\\progression\\2026-09-09_13-33-46\\frame_08_amp0.15.jpeg",
        "D:\\Github\\PhD Code\\Erythema-Progression\\output\\different_skin\\45\\progression\\2026-09-09_13-34-20\\frame_08_amp0.15.jpeg",
        "D:\\Github\\PhD Code\\Erythema-Progression\\output\\different_skin\\190\\progression\\2026-09-09_13-35-15\\frame_08_amp0.15.jpeg"
    ]
    create_image_grid(paths, cols=3, tile_size=(400, 400), output_path="final_grid.jpg")