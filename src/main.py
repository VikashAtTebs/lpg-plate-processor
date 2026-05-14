from pathlib import Path
import os
import shutil

import cv2

from detect_plate import detect_and_crop_plate
from classify_plate import classify_plate_image


INPUT_DIR = Path("input_images")
CROPPED_DIR = Path("output/cropped")
BASE_OUT = Path("sorted_output")

SERIAL_DIR = BASE_OUT / "Serial_No"
BATCH_TEST_DIR = BASE_OUT / "Batch_No_Test_Date"
BATCH_ONLY_DIR = BASE_OUT / "Batch_No"
REJECTED_DIR = BASE_OUT / "rejected"

for d in [CROPPED_DIR, SERIAL_DIR, BATCH_TEST_DIR, BATCH_ONLY_DIR, REJECTED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def save_horizontal(img_bgr, path):
    """Rotate to horizontal if needed, then save."""
    h, w = img_bgr.shape[:2]
    out = img_bgr
    if h > w:
        out = cv2.rotate(img_bgr,  cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(str(path), out)


def main():
    image_paths = (
        list(INPUT_DIR.glob("*.jpg"))
        + list(INPUT_DIR.glob("*.jpeg"))
        + list(INPUT_DIR.glob("*.png"))
    )

    if not image_paths:
        print(f"No images found in {INPUT_DIR.resolve()}")
        return

    for img_path in image_paths:
        print(f"\nProcessing {img_path.name} ...")

        # --- load original once ---
        orig = cv2.imread(str(img_path))
        if orig is None:
            print("  [WARN] Cannot read image, sending to rejected.")
            shutil.move(str(img_path), REJECTED_DIR / img_path.name)
            continue

        # --- 1) try cropped plate first ---
        cropped_path = detect_and_crop_plate(str(img_path), str(CROPPED_DIR))

        best_label = "rejected"
        best_source = None          # "crop" or "orig"
        best_image = None           # numpy image to save

        # 1a. classification on cropped plate (if we have one)
        # if cropped_path is not None and os.path.exists(cropped_path):
        #     crop_img = cv2.imread(cropped_path)
        #     if crop_img is not None:
        #         label_crop = classify_plate_image(crop_img)
        #         print("  cropped label:", label_crop)
        #         if label_crop != "rejected":
        #             best_label = label_crop
        #             best_source = "crop"
        #             best_image = crop_img

        # # --- 2) fallback: original image if cropped attempt failed ---
        # if best_label == "rejected":
        #     label_orig = classify_plate_image(orig)
        #     print("  original label:", label_orig)
        #     if label_orig != "rejected":
        #         best_label = label_orig
        #         best_source = "orig"
        #         best_image = orig

        # # --- 3) route based on final label ---
        # if best_label == "rejected" or best_image is None:
        #     print("  -> rejected")
        #     target_dir = REJECTED_DIR
        #     target_dir.mkdir(parents=True, exist_ok=True)

        #     # move original as-is to rejected
        #     shutil.move(str(img_path), target_dir / img_path.name)
        #     continue

        # # choose target folder
        # if best_label == "Serial_No":
        #     target_dir = SERIAL_DIR
        # elif best_label == "Batch_No_Test_Date":
        #     target_dir = BATCH_TEST_DIR
        # elif best_label == "Batch_No":
        #     target_dir = BATCH_ONLY_DIR
        # else:
        #     target_dir = REJECTED_DIR

        # target_dir.mkdir(parents=True, exist_ok=True)

        # # --- 4) save ONLY the image that produced this label ---

        # dest_path = target_dir / img_path.name

        # if best_source == "crop":
        #     # cropped plate gave us the label → save cropped, not original
        #     print(f"  -> {best_label} (from CROPPED), saving cropped as {dest_path.name}")
        #     save_horizontal(best_image, dest_path)
        # else:
        #     # original image gave us the label → rotate to horizontal and save
        #     print(f"  -> {best_label} (from ORIGINAL), saving rotated original as {dest_path.name}")
        #     save_horizontal(best_image, dest_path)

        # # Remove the original from input_images after sorting
        # if img_path.exists():
        #     os.remove(str(img_path))

        # # Optionally, clean up cropped file to save space
        # if cropped_path is not None and os.path.exists(cropped_path):
        #     os.remove(cropped_path)


if __name__ == "__main__":
    main()