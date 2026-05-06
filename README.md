# LPG Stay Plate Image Processing

This project automates processing of LPG cylinder stay-plate images:

- Crops each image so that only the metal stay plate is visible.
- Optionally refines the crop using Meta’s Segment Anything Model (SAM).
- (Optional) Runs OCR and fuzzy text matching to route images into folders:
  - `Serial_No`
  - `Batch_No_Test_Date`
  - `Batch_No`
  - `rejected`

The goal is to handle hundreds of images automatically with minimal manual sorting, even when text is low-contrast or partially occluded.

---

## Project Structure

```text
image-processing-workspace/
├── .venv/                  # Python virtual environment (ignored)
├── configs/
│   └── config.yaml         # (Optional) project-specific settings
├── input_images/           # Raw input images (ignored)
├── logs/                   # Log files
├── output/
│   └── cropped/            # Cropped plate images (ignored)
├── rejected/               # Images where plate cannot be detected / classified
├── src/
│   ├── detect_plate.py     # Core plate detection + cropping logic
│   ├── sam_plate.py        # SAM integration (optional, for fine segmentation)
│   ├── classify_plate.py   # OCR + fuzzy matching based classifier (optional)
│   └── main.py             # Entry point: batch processing pipeline
├── tests/                  # (Optional) test scripts
├── .gitignore
├── requirements.txt
└── README.md
```

### `.gitignore`

The following entries keep local-only artifacts out of git:

```gitignore
/output/cropped/*
/input_images/*
/.venv
```

This ensures the repo only contains code and configuration, not large binary data or environment files.[web:173]

---

## Requirements

### Python

- Python 3.10+ (project currently tested on 3.12)

### Core Python packages

These are the main runtime dependencies (see `requirements.txt` for exact versions):

- `numpy` – array operations
- `opencv-python-headless` – image loading, preprocessing, and basic computer vision
- `PyYAML` – loading configs (if used)
- `watchdog` – optional, for filesystem monitoring / auto-runs

### Optional: OCR + fuzzy matching

If you enable automatic folder classification based on plate text:

- `pytesseract` – Python wrapper for Tesseract OCR[web:10][web:158]
- `rapidfuzz` – fast fuzzy string matching for robust keyword detection[web:156][web:164]

You also need the **Tesseract OCR binary** installed on the system:

```bash
sudo apt install tesseract-ocr
```

### Optional: SAM (Segment Anything Model)

If you use SAM for segmentation-level cropping:

- `torch`, `torchvision` – PyTorch backend
- `segment-anything` – Meta’s Segment Anything implementation[web:125][web:123]

You also need a SAM checkpoint (e.g. `sam_vit_b_01ec64.pth`) downloaded into `checkpoints/`.

---

## Setup

1. **Clone the repository**

   ```bash
   git clone git@github.com:<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Install Tesseract OCR**

   ```bash
   sudo apt install tesseract-ocr
   ```

5. **(Optional) Download SAM checkpoint**

   ```bash
   mkdir -p checkpoints
   cd checkpoints
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
   cd ..
   ```

   Update `sam_plate.py` if you use a different path or model type.

---

## Usage

### 1. Prepare input images

Place all raw cylinder photos into `input_images/` (top-level jpg/png files).

Example:

```text
input_images/
├── stay_plate_1_123.jpg
├── stay_plate_1_456.jpg
└── ...
```

### 2. Run the processing pipeline

From the project root:

```bash
source .venv/bin/activate
python -m src.main
```

What `main.py` typically does:

1. Enumerates all images in `input_images/`.
2. For each image:

   - Calls `detect_plate.detect_and_crop_plate()` to:
     - Load the image.
     - Decide whether to keep full frame or center crop as a base.
     - Optionally pass the base image through `sam_plate.crop_plate_with_sam()` for fine segmentation around the stay plate.
     - Rotate the result to horizontal orientation if needed.
     - Save the cropped plate to `output/cropped/`.

   - (Optional) Calls `classify_plate.classify_plate_image()` to:
     - Preprocess the cropped plate (CLAHE, binarization) for OCR.[web:160][web:163]
     - Run Tesseract via `pytesseract` to extract text.
     - Use `rapidfuzz` to fuzzy‑match keywords such as:
       - `SR.NO`, `SERIAL NO` → Serial number plate.
       - `BATCH NO`, `TEST DATE` → Batch & test date plate.
     - Decide on one of:
       - `Serial_No`
       - `Batch_No_Test_Date`
       - `Batch_No`
       - `rejected`

   - Moves the original image (or cropped file, depending on configuration) into the corresponding folder:
     - `sorted_output/Serial_No/`
     - `sorted_output/Batch_No_Test_Date/`
     - `sorted_output/Batch_No/`
     - `sorted_output/rejected/`

3. Logs basic information (file name, plate detection status, classification label).

---

## Configuration

If `configs/config.yaml` is used, it can hold tunable parameters such as:

- Cropping fractions (width/height for center crop).
- HSV thresholds for plate color.
- SAM enable/disable flag.
- OCR / fuzzy-matching thresholds.

Example (pseudo):

```yaml
crop:
  width_frac: 0.6
  height_frac: 0.6

sam:
  enabled: true
  model_type: "vit_b"
  checkpoint: "checkpoints/sam_vit_b_01ec64.pth"

ocr:
  enabled: false  # set true when OCR is wired in
```

Update `detect_plate.py`, `sam_plate.py`, and `classify_plate.py` to read these values if you want fully configurable behavior.

---

## Development Notes

- **Dependency pinning:** Versions in `requirements.txt` are pinned (`==`) so others can reproduce your working environment exactly.[web:168][web:171]  
- **.venv ignored:** The virtual environment is not committed; each developer creates their own `.venv` and runs `pip install -r requirements.txt`.  
- **Image folders ignored:** `input_images/` and `output/cropped/` are ignored to keep the repository small and free from user data.  

---

## Roadmap / Ideas

- Improve OCR preprocessing for extremely dusty / low‑contrast plates.
- Log OCR text + predicted label for each file to a CSV for easier rule tuning.
- Replace rule‑based classification with a small trained text classifier if the dataset grows.
- Add tests in `tests/` for plate detection and classification functions.

---

## License

Add your license information here (e.g. MIT, Apache 2.0, proprietary).