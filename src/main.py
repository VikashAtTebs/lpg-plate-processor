from pathlib import Path

from detect_plate import detect_and_crop_plate


INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("output/cropped")


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
        detect_and_crop_plate(str(img_path), str(OUTPUT_DIR))


if __name__ == "__main__":
    main()