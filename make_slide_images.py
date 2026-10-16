"""
Creates clean, PPT-ready slide images using real NephroVision scan results.
Outputs to f:/Capstone_Kidney_Project/slide_images/
"""

from PIL import Image, ImageDraw, ImageFont
import os

OUT_DIR = "f:/Capstone_Kidney_Project/slide_images"
os.makedirs(OUT_DIR, exist_ok=True)

# ── colour palette ──────────────────────────────────────────────
NAVY   = (10,  31,  68)
BLUE   = (0,  116, 217)
ORANGE = (255, 107,  53)
WHITE  = (255, 255, 255)
LGREY  = (245, 247, 250)
GREEN  = (0,  180, 100)
RED    = (220,  50,  50)

def load_crop(path, size=(460, 380)):
    """Load image, crop center square, resize."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    # crop one scan from the grid (top-left cell)
    cell_w, cell_h = w // 4, h // 4
    img = img.crop((0, 0, cell_w * 2, cell_h * 2))
    img = img.resize(size, Image.LANCZOS)
    return img

def draw_rounded_rect(draw, xy, radius, fill):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill)

def get_font(size, bold=False):
    """Try to load a font, fall back to default."""
    font_paths = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except:
                pass
    return ImageFont.load_default()


# ════════════════════════════════════════════════════════════════
# IMAGE 1 — "What AI Detects" — 3 real scans side by side
# ════════════════════════════════════════════════════════════════
def make_detection_panel():
    W, H = 1400, 600
    img = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)

    # Header banner
    draw.rectangle([0, 0, W, 80], fill=NAVY)
    draw.rectangle([0, 76, W, 84], fill=ORANGE)
    f_title = get_font(32, bold=True)
    draw.text((30, 18), "NephroVision — Real AI Detection Results", font=f_title, fill=WHITE)

    # 3 panels
    panels = [
        {
            "img_path": "f:/Capstone_Kidney_Project/YOLO_Segmentation/cyst_segmentation_v1/val_batch0_pred.jpg",
            "label": "CYST DETECTED",
            "desc": "Blue box = fluid-filled sac\nAI confidence: 90%",
            "color": BLUE,
            "badge_bg": (0, 100, 200),
        },
        {
            "img_path": "f:/Capstone_Kidney_Project/YOLO_Segmentation/tumor_segmentation_v1/val_batch0_pred.jpg",
            "label": "TUMOR DETECTED",
            "desc": "Cyan box = abnormal growth\nAI confidence: 95-100%",
            "color": RED,
            "badge_bg": (180, 30, 30),
        },
        {
            "img_path": "f:/Capstone_Kidney_Project/YOLO_Segmentation/unified_segmentation_v1/val_batch0_pred.jpg",
            "label": "UNIFIED DETECTION",
            "desc": "Detects Tumor + Cyst\nin same scan",
            "color": (0, 150, 80),
            "badge_bg": (0, 130, 70),
        },
    ]

    f_label = get_font(22, bold=True)
    f_desc  = get_font(17)
    f_badge = get_font(16, bold=True)

    for i, p in enumerate(panels):
        x = 30 + i * 460
        # scan image
        try:
            scan = load_crop(p["img_path"], size=(420, 340))
            img.paste(scan, (x, 95))
        except Exception as e:
            draw.rectangle([x, 95, x+420, 435], fill=(30,30,30))
            draw.text((x+140, 250), "[scan]", font=f_label, fill=WHITE)

        # coloured label badge
        bw = 200
        draw.rounded_rectangle([x, 445, x + bw, 480], radius=8, fill=p["badge_bg"])
        draw.text((x + 10, 450), p["label"], font=f_badge, fill=WHITE)

        # description
        for j, line in enumerate(p["desc"].split("\n")):
            draw.text((x, 492 + j*26), line, font=f_desc, fill=(60, 60, 80))

        # thin left accent bar
        draw.rectangle([x - 5, 95, x, 440], fill=p["color"])

    img.save(f"{OUT_DIR}/slide_detection_panel.png", quality=95)
    print("Saved: slide_detection_panel.png")


# ════════════════════════════════════════════════════════════════
# IMAGE 2 — Simple 6-Step Flow (horizontal)
# ════════════════════════════════════════════════════════════════
def make_flow_diagram():
    W, H = 1400, 340
    img = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, W, 60], fill=NAVY)
    draw.rectangle([0, 57, W, 64], fill=ORANGE)
    f_title = get_font(26, bold=True)
    draw.text((30, 13), "How NephroVision Works — 6 Simple Steps", font=f_title, fill=WHITE)

    steps = [
        ("1", "Upload\nCT Scan",       BLUE,              "Doctor / Nurse"),
        ("2", "AI Reads\nthe Scan",     (60,  60, 160),    "Automatic"),
        ("3", "Problem\nHighlighted",   RED,               "On Screen"),
        ("4", "Kidney\nMeasured",       (130,  60, 180),   "Both sides"),
        ("5", "Ask AI\nQuestions",      (0,  140,  90),    "Instant answer"),
        ("6", "Download\nReport",       ORANGE,            "PDF ready"),
    ]

    f_num  = get_font(30, bold=True)
    f_step = get_font(18, bold=True)
    f_sub  = get_font(15)

    box_w = 190
    gap   = 22
    start = 20

    for i, (num, label, col, sub) in enumerate(steps):
        x = start + i * (box_w + gap)
        # box
        draw.rounded_rectangle([x, 78, x + box_w, 260], radius=14, fill=col)
        # number circle
        draw.ellipse([x + 12, 88, x + 56, 132], fill=WHITE)
        draw.text((x + 20, 92), num, font=f_num, fill=col)
        # label
        for j, line in enumerate(label.split("\n")):
            draw.text((x + 14, 145 + j * 28), line, font=f_step, fill=WHITE)
        # sub
        draw.text((x + 14, 216), sub, font=f_sub, fill=(220, 235, 255))

        # arrow
        if i < len(steps) - 1:
            ax = x + box_w + 4
            draw.polygon([(ax, 162), (ax + gap - 4, 169), (ax, 176)], fill=NAVY)

    # bottom note
    f_note = get_font(17, bold=True)
    draw.rounded_rectangle([200, 275, 1200, 320], radius=10, fill=(255, 247, 230))
    draw.rectangle([200, 275, 210, 320], fill=ORANGE)
    draw.text((225, 285),
              "Total time from scan upload to full report:  Under 2 minutes",
              font=f_note, fill=NAVY)

    img.save(f"{OUT_DIR}/slide_flow_diagram.png", quality=95)
    print("Saved: slide_flow_diagram.png")


# ════════════════════════════════════════════════════════════════
# IMAGE 3 — Metrics Dashboard
# ════════════════════════════════════════════════════════════════
def make_metrics_panel():
    W, H = 1400, 420
    img = Image.new("RGB", (W, H), LGREY)
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, W, 70], fill=NAVY)
    draw.rectangle([0, 67, W, 74], fill=ORANGE)
    f_title = get_font(28, bold=True)
    draw.text((30, 16), "NephroVision — Proven Performance", font=f_title, fill=WHITE)

    metrics = [
        ("95.4%",  "Accuracy",             "Correctly identifies\nkidney conditions",   BLUE),
        ("95.3%",  "F1 Score",             "Reliable across all\n4 conditions",          (60, 60, 160)),
        ("93.8%",  "Detection Rate",       "Finds the exact\naffected area",             GREEN),
        ("< 2 sec","Speed",                "Full analysis\nper patient",                 ORANGE),
        ("10,000+","Scans Trained On",     "Real CT images\nused to train AI",           RED),
    ]

    f_big  = get_font(36, bold=True)
    f_med  = get_font(18, bold=True)
    f_sm   = get_font(15)

    card_w = 245
    gap    = 18
    sx     = 30

    for i, (val, lbl, desc, col) in enumerate(metrics):
        x = sx + i * (card_w + gap)
        # card
        draw.rounded_rectangle([x, 88, x + card_w, 380], radius=16, fill=WHITE)
        # top accent
        draw.rounded_rectangle([x, 88, x + card_w, 106], radius=16, fill=col)
        draw.rectangle([x, 100, x + card_w, 106], fill=col)
        # value
        draw.text((x + 18, 120), val, font=f_big, fill=col)
        # label
        draw.text((x + 18, 186), lbl, font=f_med, fill=NAVY)
        # desc
        for j, line in enumerate(desc.split("\n")):
            draw.text((x + 18, 222 + j * 26), line, font=f_sm, fill=(80, 80, 100))

    img.save(f"{OUT_DIR}/slide_metrics.png", quality=95)
    print("Saved: slide_metrics.png")


# ── Run all ────────────────────────────────────────────────────
make_detection_panel()
make_flow_diagram()
make_metrics_panel()
print("\nAll slide images saved to:", OUT_DIR)
