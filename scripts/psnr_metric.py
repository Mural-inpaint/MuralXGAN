import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from src.metrics import PSNR, SSIM

# Define paths for input images and output merged images
images_path = "../checkpoints/test/merged_output_sota"
# images_path = "../checkpoints/test/merged_output_zhixin20"
outputs_merged_path = "/mnt/d/Edinburgh/MLP/MuralDH/test-dataset-jingyang/test_dataset/images"


# Function to load images
def load_images(path, device="cuda"):
    img_list = []
    img_files = sorted([f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts images to [0, 1] range
    ])

    for img_file in img_files:
        img = Image.open(os.path.join(path, img_file))
        img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension
        img_list.append(img_tensor)

    return torch.cat(img_list, dim=0) if img_list else None


# Function for postprocessing
def postprocess(img):
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)  # B, C, H, W -> B, H, W, C
    return img.int()


# Load images and outputs_merged
images = load_images(images_path)
outputs_merged = load_images(outputs_merged_path)

# Make sure we have valid data
if images is None or outputs_merged is None:
    raise ValueError("No images found in the specified directories")

# Ensure the same shape
if images.shape != outputs_merged.shape:
    raise ValueError(f"Shape mismatch: images {images.shape} vs outputs_merged {outputs_merged.shape}")

# Calculate metrics
psnr_metric = PSNR(255.0).to("cuda")
psnr = psnr_metric(postprocess(images), postprocess(outputs_merged))
mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()

# 计算SSIM (注意：SSIM期望输入范围为[0, 1])
ssim_metric = SSIM().to("cuda")
ssim = ssim_metric(images, outputs_merged)

print(f"PSNR: {psnr.item():.4f}")
print(f"MAE: {mae.item():.4f}")
print(f"SSIM: {ssim.item():.4f}")