import cv2
import numpy as np
import imutils
import re
import os
import csv  # ✅ THÊM

from fast_plate_ocr import LicensePlateRecognizer

# ========================
# OCR SETUP
# ========================
m = LicensePlateRecognizer('cct-s-v2-global-model')

# ========================
# VIETNAMESE PLATE VALIDATOR
# ========================
VN_PLATE_PATTERNS = [
    r'^\d{2}[A-Z]{1,2}-\d{3}\.\d{2}$',
    r'^\d{2}[A-Z]\d-\d{3}\.\d{2}$',
    r'^\d{2}[A-Z]{1,2}\d{5}$',
    r'^\d{2}[A-Z]\d\d{5}$',
    r'^\d{5}\d[A-Z]{2}$',
    r'^\d{5}[A-Z]{2}$',
    r'^\d{0}[A-Z]{1,2}\d{2}-\d{2}$',
    r'^\d{5}[A-Z]{1,2}\d{1}$',
    r'^\d{0}[A-Z]{2}\d{2}\d{2}$',
    r'^\d{2}[A-Z]{1,2}\d{4}$',
]

def is_valid_vn_plate(plate: str) -> bool:
    if not plate or plate == "N/A":
        return False
    plate = plate.strip().upper()
    for pattern in VN_PLATE_PATTERNS:
        if re.match(pattern, plate):
            return True
    return False

def format_vn_plate(plate: str) -> str:
    plate = plate.strip().upper()
    if re.match(r'^\d{2}[A-Z]{1,2}-\d{3}\.\d{2}$', plate):
        return plate
    if re.match(r'^\d{2}[A-Z]\d-\d{3}\.\d{2}$', plate):
        return plate
    m = re.match(r'^(\d{2}[A-Z]{1,2})(\d{3})(\d{2})$', plate)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    m = re.match(r'^(\d{2}[A-Z]\d)(\d{3})(\d{2})$', plate)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    return plate

def ocr_candidate(plate_img) -> str:
    tmp_path = "_tmp_plate.jpg"
    cv2.imwrite(tmp_path, plate_img)
    results = m.run(tmp_path)
    if results and results[0].plate:
        return format_vn_plate(results[0].plate)
    return "N/A"

# ========================
# UTILITY: IoU + NMS
# ========================
def maximizeContrast(imgGrayscale):
    height, width = imgGrayscale.shape
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=10)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=10)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(cv2.add(imgGrayscale, imgTopHat), imgBlackHat)
    return imgGrayscalePlusTopHatMinusBlackHat

def iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    inter = ix * iy
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0

def nms(boxes, iou_thresh=0.3):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept, used = [], [False] * len(boxes)
    for i, b1 in enumerate(boxes):
        if used[i]:
            continue
        kept.append(b1)
        for j in range(i + 1, len(boxes)):
            if not used[j] and iou(b1, boxes[j]) > iou_thresh:
                used[j] = True
    return kept

# ========================
# METHOD 1: Contour + Canny
# ========================
def candidates_contour(gray):
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        area = w * h
        ratio = w / float(h)
        if (4.0 < ratio < 5.0 and 4000 < area) or \
           (1 < ratio < 2 and 4000 < area):
            boxes.append((x, y, w, h))
    return boxes

# ========================
# METHOD 2: MSER
# ========================
def candidates_mser(gray):
    mser = cv2.MSER_create(delta=5, min_area=60, max_area=50000)
    regions, _ = mser.detectRegions(gray)

    boxes = []
    for region in regions:
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        x, y, w, h = cv2.boundingRect(hull)
        if h == 0:
            continue
        ratio = w / float(h)
        if (2.0 < ratio < 5.0 and w > 60 and h > 20) or \
           (1.2 < ratio < 2.0 and w > 60 and h > 40):
            boxes.append((x, y, w, h))
    return boxes

# ========================
# COMBINED DETECTOR
# ========================
def detect_candidates(img, gray, max_candidates=20):
    contour_boxes = candidates_contour(gray)
    mser_boxes    = candidates_mser(gray)

    tagged = [("contour", b) for b in contour_boxes] + \
             [("mser",    b) for b in mser_boxes]

    kept_boxes = nms([b for _, b in tagged], iou_thresh=0.3)

    def find_tag(box):
        for tag, b in tagged:
            if b == box:
                return tag
        return "unknown"

    candidates = []
    for box in kept_boxes[:max_candidates]:
        x, y, w, h = box
        plate = img[y:y+h, x:x+w]
        if plate.size == 0:
            continue
        candidates.append((x, y, w, h, plate, find_tag(box)))

    return candidates

# ========================
# BATCH PROCESS ALL IMAGES
# ========================
folder = r"sumdoc"
images = os.listdir(folder)

success_files = []
fail_files = []

results_table = []  # ✅ THÊM: lưu kết quả CSV

for img_name in images:

    # print("\n==============================")
    # print("Processing:", img_name)
    # print("==============================")

    path = os.path.join(folder, img_name)
    img = cv2.imread(path)

    if img is None:
        print("Cannot read:", img_name)
        fail_files.append(img_name)
        results_table.append((img_name, "Cannot read image"))  # ✅ THÊM
        continue

    img = imutils.resize(img, width=800)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 40, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    rect = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    mcont = maximizeContrast(blur)
    imgThresh = cv2.adaptiveThreshold(
        mcont, 255.0,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        19, 9
    )

    candidates = detect_candidates(imgThresh, gray)

    display = img.copy()
    found_texts = []  # ✅ THÊM

    for i, (x, y, w, h, plate, source) in enumerate(candidates):
        text = ocr_candidate(plate)

        if not is_valid_vn_plate(text):
            continue

        found_texts.append(text)  # ✅ THÊM
        # print(f"Candidate {i} [{source}]: {text}")

        color = (0, 255, 0) if source == "contour" else (255, 0, 0)
        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
        cv2.putText(display, text, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if found_texts:
        success_files.append(img_name)
        combined_text = "||".join(found_texts)  # ✅ THÊM
    else:
        # print("No valid plate found.")
        fail_files.append(img_name)
        combined_text = "No valid plate found"  # ✅ THÊM

    results_table.append((img_name, f" {combined_text}"))  # ✅ THÊM

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ========================
# SUMMARY
# ========================
print("\n========== OCR SUMMARY ==========")
print(f"Images with valid plates: {len(success_files)}")
print(f"Images with NO valid plates: {len(fail_files)}")
print(f"Total images: {len(images)}")

# ========================
# EXPORT CSV
# ========================
csv_file = "license_plate_results.csv"

with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Image Name", "Detected Plates"])
    writer.writerows(results_table)

print(f"\nSaved CSV to: {csv_file}")