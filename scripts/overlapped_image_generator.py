from PIL import Image
import yaml
import os

def overlay_mask_on_image(original_path, mask_path, color=(255, 0, 0), alpha=180):
    # 1. Load original image in RGB mode
    img = Image.open(original_path).convert("RGB")

    # 2. Convert to RGBA for the compositing step
    img_rgba = img.convert("RGBA")

    # 3. Create a grayscale mask
    mask = Image.open(mask_path).convert("L")

    # 4. Create an RGB overlay image with your color
    overlay_rgb = Image.new("RGB", mask.size, color=color)

    # 5. Convert that overlay to RGBA, applying the desired alpha
    overlay_rgba = overlay_rgb.convert("RGBA")
    # (We can set alpha per pixel using our grayscale mask.)
    alpha_mask = mask.point(lambda p: p * (alpha / 255))
    overlay_rgba.putalpha(alpha_mask)

    # 6. Alpha-composite the overlay onto the RGBA image
    composite_rgba = Image.alpha_composite(img_rgba, overlay_rgba)

    # 7. Convert back to RGB (so the final image has no alpha channel)
    composite_rgb = composite_rgba.convert("RGB")

    return composite_rgb

# Example usage
config_path = "../checkpoints/config.yml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

image_flist_path = config.get("TRAIN_FLIST")
mask_flist_path = config.get("TRAIN_MASK_FLIST")

with open(image_flist_path, "r") as f:
    image_paths = [line.strip() for line in f.readlines()]

with open(mask_flist_path, "r") as f:
    mask_paths = [line.strip() for line in f.readlines()]

if len(image_paths) != len(mask_paths):
    print("Warning: Image and mask file lists have different lengths!")

output_dir = "/mnt/d/Edinburgh/MLP/MuralDH/Mural_overlay"
os.makedirs(output_dir, exist_ok=True)

for image_path, mask_path in zip(image_paths, mask_paths):
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"Skipping {image_path} or {mask_path}: File not found.")
        continue

    overlayed_image = overlay_mask_on_image(image_path, mask_path)

    filename = os.path.basename(image_path)
    overlayed_path = os.path.join(output_dir, filename)

    overlayed_image.save(overlayed_path, format="PNG")
    print(f"Saved overlay image: {overlayed_path}")