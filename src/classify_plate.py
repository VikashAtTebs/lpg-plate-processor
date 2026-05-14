import cv2
import numpy as np
import pytesseract
from rapidfuzz import fuzz
import re

def has_clear_inscription(text):
    import re

    tl = text.lower()
    compact = tl.replace(" ", "")

    cleaned = re.sub(r"[^a-z0-9./ -]", " ", tl)
    tokens = [t for t in cleaned.split() if len(t) >= 2]

    digit_tokens = [t for t in tokens if any(c.isdigit() for c in t)]
    alpha_tokens = [t for t in tokens if any(c.isalpha() for c in t)]

    plate_cues = 0
    cues = [
        "sr.no", "sr no", "srno", "serial",
        "batch", "test", "date",
        "gross", "tare", "wt", "kg"
    ]

    for cue in cues:
        if cue in tl or cue in compact:
            plate_cues += 1

    # Accept if there is enough meaningful OCR evidence
    if plate_cues >= 2 and len(tokens) >= 4:
        return True

    if plate_cues >= 1 and len(digit_tokens) >= 2 and len(alpha_tokens) >= 2:
        return True

    return False

def best_score(text, patterns):
    """Return max fuzzy partial_ratio between text and any pattern."""
    text = text.lower()
    best = 0
    for p in patterns:
        score = fuzz.partial_ratio(text, p.lower())
        if score > best:
            best = score
    return best

# If tesseract is not in PATH, set this:
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

def extract_focus_roi(img_bgr):
    """Take already-cropped plate and keep the central band where text is."""
    h, w = img_bgr.shape[:2]

    x1 = int(0.08 * w)
    x2 = int(0.92 * w)
    y1 = int(0.18 * h)
    y2 = int(0.82 * h)

    roi = img_bgr[y1:y2, x1:x2]
    return roi

def generate_ocr_variants(img_bgr):
    """
    Given an already-cropped plate, return a list of preprocessed
    grayscale/binary images to try with Tesseract.
    """
    roi = extract_focus_roi(img_bgr)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    variants = []

    for scale in (1.5, 2.0):
        g = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        g = clahe.apply(g)

        g_blur = cv2.medianBlur(g, 3)

        thr = cv2.adaptiveThreshold(
            g_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            5,
        )
        variants.append(thr)
        variants.append(cv2.bitwise_not(thr))

        kernel = np.ones((3, 3), np.uint8)
        grad = cv2.morphologyEx(g_blur, cv2.MORPH_GRADIENT, kernel)
        _, grad_bin = cv2.threshold(
            grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        variants.append(grad_bin)

    return variants

def ocr_text_multi(img_bgr):
    """
    Run Tesseract on multiple preprocessed variants and combine the text.
    """
    variants = generate_ocr_variants(img_bgr)
    texts = []

    configs = ["--psm 6 --oem 3", "--psm 7 --oem 3"]

    for var in variants:
        for cfg in configs:
            t = pytesseract.image_to_string(var, config=cfg)
            if t.strip():
                texts.append(t)

    combined = "\n".join(texts)
    return combined

def preprocess_for_ocr(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Contrast-limited adaptive histogram equalization (helps low-contrast embossed text)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Slight blur + Otsu binarization
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh


# def ocr_text(img_bgr):
#     pre = preprocess_for_ocr(img_bgr)
#     # PSM 6 = assume a block of text; OEM 3 = default LSTM engine
#     config = "--psm 6 --oem 3"
#     text = pytesseract.image_to_string(pre, config=config)
#     return text


def has_fuzzy(text, patterns, threshold=80):
    text = text.lower()
    for p in patterns:
        if fuzz.partial_ratio(text, p.lower()) >= threshold:
            return True
    return False


def classify_plate_image(img_bgr):
    text = ocr_text_multi(img_bgr)
    tl = text.lower()

    # Reject unclear / unreadable plates
    if not has_clear_inscription(text):
        return "rejected"

    # Hard SR rule
    compact = tl.replace(" ", "")
    if (
        "sr.no" in tl
        or "sr no" in tl
        or "sr.no." in tl
        or "srno" in compact
    ):
        return "Serial_No"

    serial_patterns = [
        "sr.no", "sr.no.", "sr. no", "sr no", "sr-no", "srno",
        "s r.no", "s r no",
        "serial no", "serial no."
    ]

    batch_patterns = [
        "batch no", "batch no.", "batchno", "batchno.",
        "b no", "b. no", "bno",
        "batch", "batc no", "tch no", "atch no",
    ]

    test_patterns = [
        "test date", "test date.", "test dt", "testdt", "testdate",
        "tst date", "tst dt", "test dte", "t. date", "st date",
    ]

    serial_score = best_score(tl, serial_patterns)
    batch_score = best_score(tl, batch_patterns)
    test_score = best_score(tl, test_patterns)

    S_MIN = 70
    B_MIN = 70
    T_MIN = 65

    if batch_score >= B_MIN and test_score >= T_MIN:
        return "Batch_No_Test_Date"

    if batch_score >= B_MIN and serial_score < S_MIN - 5 and test_score < T_MIN:
        return "Batch_No"

    if serial_score >= S_MIN and batch_score < B_MIN - 5:
        return "Serial_No"

    if serial_score >= S_MIN or batch_score >= B_MIN:
        if batch_score >= serial_score and test_score >= T_MIN - 5:
            if test_score >= T_MIN:
                return "Batch_No_Test_Date"
            else:
                return "Batch_No"
        else:
            return "Serial_No"

    return "rejected"