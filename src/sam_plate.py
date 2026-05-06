import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# Load SAM once at import time
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = "checkpoints/sam_vit_b_01ec64.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

# Automatic mask generator over whole image
mask_generator = SamAutomaticMaskGenerator(
    sam,
    pred_iou_thresh=0.90,       # only high‑quality masks
    stability_score_thresh=0.92,
    min_mask_region_area=3000,  # ignore tiny blobs (pixels)
)


def crop_plate_with_sam(image_bgr):
    """
    Given a BGR image (full or center-crop),
    run SAM, pick the mask that looks like the plate,
    and return a refined crop. If nothing good is found,
    return the original image.
    """
    h, w = image_bgr.shape[:2]

    # SAM expects RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image_rgb)  # list of dicts[web:126][web:133]
    if not masks:
        return image_bgr

    best = None
    best_score = 0.0

    for m in masks:
        area = m["area"]
        x, y, ww, hh = m["bbox"]  # [x, y, w, h] in pixels
        iou = m.get("predicted_iou", 1.0)

        # filter by size
        if area < 0.05 * h * w:
            continue

        # filter by elongated rectangle shape (plate-like)
        aspect = max(ww, hh) / float(min(ww, hh) + 1e-6)
        if aspect < 1.2 or aspect > 6.0:
            continue

        score = area * iou
        if score > best_score:
            best_score = score
            best = (x, y, ww, hh)

    if best is None:
        # fall back to original
        return image_bgr

    x, y, ww, hh = best

    # SAM bboxes are floats; convert to ints
    x = int(round(x))
    y = int(round(y))
    ww = int(round(ww))
    hh = int(round(hh))

    pad = int(0.03 * max(ww, hh))

    x1 = int(max(0, x - pad))
    y1 = int(max(0, y - pad))
    x2 = int(min(w, x + ww + pad))
    y2 = int(min(h, y + hh + pad))

    crop = image_bgr[y1:y2, x1:x2]

    return crop