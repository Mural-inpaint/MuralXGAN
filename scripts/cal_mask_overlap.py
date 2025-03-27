import random
import numpy as np
import cv2

flist_path = "/mnt/d/Edinburgh/MLP/MuralDH/Mural_seg/train/labels.flist"
img_height, img_width = 256, 256

def load_flist(flist_path):
    with open(flist_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]


def load_and_binarize_mask(path, h, w):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask at {path}")
    mask = cv2.resize(mask, (w, h))
    mask = (mask > 0).astype(np.uint8)  # 0 或 1
    return mask


def compute_overlap(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


mask_paths = load_flist(flist_path)


original_masks = [load_and_binarize_mask(p, img_height, img_width) for p in mask_paths]


shuffled_paths = mask_paths.copy()
random.shuffle(shuffled_paths)
shuffled_masks = [load_and_binarize_mask(p, img_height, img_width) for p in shuffled_paths]


overlaps = [compute_overlap(m1, m2) for m1, m2 in zip(original_masks, shuffled_masks)]


print(f"Average IoU between original and shuffled masks: {np.mean(overlaps):.4f}")
