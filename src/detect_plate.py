import cv2
import os
from sam_plate import crop_plate_with_sam


def detect_and_crop_plate(image_path: str, output_dir: str):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Could not read image: {image_path}")
        return None

    img_h, img_w = img.shape[:2]

    # 0) same “mostly plate” check as before (optional)
    # ... your HSV ratio logic here ...

    # 1) your current center-crop / no-crop logic → base
    #    (this keeps your robust heuristics)
    base = img  # or your previous base = center_crop(img)

    # 2) Let SAM refine to exact plate region
    refined = crop_plate_with_sam(base)

    # 3) Ensure horizontal orientation
    crop = refined
    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, crop)

    print(f"[OK] Saved SAM-cropped plate: {out_path}")
    return out_path