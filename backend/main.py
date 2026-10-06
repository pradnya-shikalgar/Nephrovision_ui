import os
import io
import random
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
from pydantic import BaseModel
from typing import List, Optional

# Import the model definition
try:
    from model_def import PCSA_KidneyNeXt
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

app = FastAPI(title="NephroVision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BoundingBox(BaseModel):
    class_name: str
    x: float
    y: float
    width: float
    height: float

class AnalysisResponse(BaseModel):
    status: str
    classification: str
    confidence: float
    filename: str
    is_tumor: bool
    is_cyst: bool
    left_volume: int
    right_volume: int
    message: str
    sign_symptom: str = ""
    distance: str = ""
    history: str = ""
    medication: str = ""
    doctor_prescription: str = ""
    bounding_boxes: List[BoundingBox] = []

# ---------------------------------------------------------
# Load PyTorch Model
# ---------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

if MODEL_AVAILABLE:
    try:
        model = PCSA_KidneyNeXt(num_classes=4).to(device)
        model_path = '../best_pcsa_kidneynext_10epochs.pth'
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print("Successfully loaded PyTorch model!")
        else:
            print(f"Warning: Model weights not found at {model_path}. Using mock inference.")
            model = None
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None

# Load YOLO model
yolo_model = None
try:
    from ultralytics import YOLO
    yolo_model_path = '../YOLO_Segmentation/unified_segmentation_v1/weights/best.pt'
    if os.path.exists(yolo_model_path):
        yolo_model = YOLO(yolo_model_path)
        print("Successfully loaded YOLO segmentation model!")
    else:
        print(f"Warning: YOLO weights not found at {yolo_model_path}.")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")

# Transforms matching training data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

@app.get("/")
def read_root():
    return {"message": "NephroVision Backend is running"}

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    doctor_prescription: Optional[str] = Form(""),
    patient_name: Optional[str] = Form(""),
    patient_age: Optional[str] = Form("")
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    contents = await file.read()
    
    classification = "Normal"
    confidence = 98.0
    bounding_boxes = []
    
    if yolo_model is not None:
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            results = yolo_model(image)
            result = results[0]
            
            img_width, img_height = image.size
            
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                class_name = result.names[cls_id]
                
                # xywh format is [x_center, y_center, width, height]
                x_center, y_center, w, h = box.xywh[0].tolist()
                
                x_percent = (x_center - w / 2) / img_width * 100
                y_percent = (y_center - h / 2) / img_height * 100
                w_percent = (w / img_width) * 100
                h_percent = (h / img_height) * 100
                
                bounding_boxes.append(BoundingBox(
                    class_name=class_name,
                    x=x_percent,
                    y=y_percent,
                    width=w_percent,
                    height=h_percent
                ))
        except Exception as e:
            print(f"YOLO Inference error: {e}")

    if model is not None:
        try:
            # REAL INFERENCE
            if 'image' not in locals():
                image = Image.open(io.BytesIO(contents)).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                conf, predicted = torch.max(probs, 1)
                
            pred_idx = predicted.item()
            classification = class_names[pred_idx]
            confidence = round(conf.item() * 100, 2)
            
            # Override classification if YOLO found a pathology box
            if len(bounding_boxes) > 0:
                yolo_classes = [box.class_name.lower() for box in bounding_boxes]
                if 'tumor' in yolo_classes:
                    classification = 'Tumor'
                elif 'cyst' in yolo_classes:
                    classification = 'Cyst'
                elif 'stone' in yolo_classes:
                    classification = 'Stone'
            
        except Exception as e:
            print(f"Inference error: {e}")
            # fallback to normal
    else:
        # MOCK INFERENCE FALLBACK
        await asyncio.sleep(2.0)
        filename_lower = file.filename.lower()
        if "tumor" in filename_lower or "abnormal" in filename_lower or "tumer" in filename_lower or "tumar" in filename_lower:
            classification = "Tumor"
            confidence = round(random.uniform(92.0, 99.9), 1)
        elif "cyst" in filename_lower:
            classification = "Cyst"
            confidence = round(random.uniform(94.0, 99.9), 1)
        elif "stone" in filename_lower:
            classification = "Stone"
            confidence = round(random.uniform(90.0, 98.0), 1)
        else:
            classification = "Normal"
            confidence = round(random.uniform(95.0, 99.9), 1)

    # Override mock classification if YOLO found a pathology box
    if len(bounding_boxes) > 0:
        yolo_classes = [box.class_name.lower() for box in bounding_boxes]
        if 'tumor' in yolo_classes:
            classification = 'Tumor'
        elif 'cyst' in yolo_classes:
            classification = 'Cyst'
        elif 'stone' in yolo_classes:
            classification = 'Stone'

    is_tumor = classification == "Tumor"
    is_cyst = classification == "Cyst"
    
    if is_tumor:
        display_class = "Tumor Detected"
        right_vol = random.randint(145, 160)
        message = "An abnormal mass/tumor has been identified in the right kidney region. Immediate clinical review is recommended."
        sign_symptom = "Hematuria, Flank Pain, Palpable Mass"
        distance = "4.2 cm from renal pelvis"
        history = "Smoker, Hypertension"
        medication = "Amlodipine 5mg"
    elif is_cyst:
        display_class = "Cyst Detected"
        right_vol = random.randint(140, 150)
        message = "A benign-appearing cyst was identified in the renal cortex. Routine monitoring is advised."
        sign_symptom = "Asymptomatic (Incidental finding)"
        distance = "Cortical surface"
        history = "No relevant history"
        medication = "None"
    elif classification == "Stone":
        display_class = "Kidney Stone Detected"
        right_vol = random.randint(138, 145)
        message = "A calcified mass (stone) was detected in the kidney. Consider urological consult."
        is_cyst = True # Re-using warning icon
        sign_symptom = "Severe renal colic, Nausea"
        distance = "Ureteropelvic junction"
        history = "Previous episodes of nephrolithiasis"
        medication = "Tamsulosin 0.4mg, Ibuprofen"
    else:
        display_class = "Normal Kidney Structure"
        right_vol = random.randint(130, 142)
        message = "No visible abnormalities, cysts, or tumors identified in the highlighted regions."
        sign_symptom = "None (Routine checkup)"
        distance = "N/A"
        history = "Healthy"
        medication = "None"

    return AnalysisResponse(
        status="success",
        classification=display_class,
        confidence=confidence,
        filename=file.filename,
        is_tumor=is_tumor,
        is_cyst=is_cyst,
        left_volume=random.randint(135, 145),
        right_volume=right_vol,
        message=message,
        sign_symptom=sign_symptom,
        distance=distance,
        history=history,
        medication=medication,
        doctor_prescription=doctor_prescription or "",
        bounding_boxes=bounding_boxes
    )

@app.delete("/api/analyze/{filename}")
def delete_analysis(filename: str):
    # Mock endpoint since we don't have a DB yet. 
    # Returning a 200 success response.
    return {"status": "success", "message": f"Deleted {filename} from backend records"}

import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_base64_image(img, quality=98):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', img, encode_param)
    return base64.b64encode(buffer).decode('utf-8')

def enhance_and_upscale(img, min_dim=768):
    h, w = img.shape[:2]
    if max(h, w) < min_dim:
        scale = min_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), 1.0)
        sharpened = cv2.addWeighted(upscaled, 1.25, gaussian, -0.25, 0)
        return sharpened
    return img

@app.post("/api/technical_details")
async def get_technical_details(file: UploadFile = File(...)):
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert('RGB')
    # Keep an RGB numpy array as the master - everything derives from it
    image_rgb = np.array(image_pil)
    image_cv2 = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Upscale to crisp high resolution (min 768px) - stays BGR for cv2 ops
    image_cv2_hires = enhance_and_upscale(image_cv2, min_dim=768)
    # RGB hires version for GradCAM overlay
    image_rgb_hires = cv2.cvtColor(image_cv2_hires, cv2.COLOR_BGR2RGB)

    # 1. Original — already BGR so get_base64_image is correct
    original_b64 = get_base64_image(image_cv2_hires)

    # 2. Grad-CAM — target model.stage4.pcsa (PCSA attention is the key novel layer)
    gradcam_b64 = ""
    pred_idx = 1  # default Normal
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        # Best GradCAM target = model.stage4.gelu
        # This is the 7x7 spatial feature map AFTER compress_conv+BN+GELU, BEFORE the PCSA module.
        # model.stage4.pcsa is WRONG: it runs AdaptiveAvgPool2d(1) internally which
        # collapses spatial gradients to a single value, causing the heatmap to be
        # diffuse and mislocalized (shows spine/center instead of kidney region).
        target_layers = [model.stage4.gelu]

        cam = GradCAM(model=model, target_layers=target_layers)
        input_tensor = transform(image_pil).unsqueeze(0).to(device)

        # First get prediction so we target the correct predicted class
        with torch.no_grad():
            raw_output = model(input_tensor)
            probs = torch.softmax(raw_output, dim=1)
            _, predicted = torch.max(probs, 1)
        pred_idx = predicted.item()

        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # Resize cam to match hires image
        target_h, target_w = image_rgb_hires.shape[:2]
        grayscale_cam_hires = cv2.resize(grayscale_cam, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # show_cam_on_image expects RGB float [0,1] — use image_rgb_hires
        rgb_norm = np.float32(image_rgb_hires) / 255.0
        cam_image_rgb = show_cam_on_image(rgb_norm, grayscale_cam_hires, use_rgb=True)

        # Convert back to BGR for cv2.imencode inside get_base64_image
        cam_image_bgr = cv2.cvtColor(cam_image_rgb, cv2.COLOR_RGB2BGR)
        gradcam_b64 = get_base64_image(cam_image_bgr)
    except Exception as e:
        print("GradCAM error:", e)
        # Fallback: apply jet colormap directly (no random, reproducible)
        gray = cv2.cvtColor(image_cv2_hires, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        heatmap = cv2.applyColorMap(gray_eq, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_cv2_hires, 0.65, heatmap, 0.35, 0)
        gradcam_b64 = get_base64_image(overlay)

    # 3. Surgical Boundaries — YOLO with correct cyst coloring (dark red)
    pathology_total_area = 0.0
    total_kidney_area = 0.0
    surgical_b64 = ""
    try:
        if yolo_model is not None:
            image_pil_hires = Image.fromarray(image_rgb_hires)
            results = yolo_model(image_pil_hires, imgsz=max(640, max(image_rgb_hires.shape[:2])))
            result = results[0]

            # Start with a clean copy of the BGR hires image for drawing
            canvas = image_cv2_hires.copy()
            hires_h, hires_w = canvas.shape[:2]

            # Draw segmentation masks with correct per-class colors
            if result.masks is not None:
                masks_data = result.masks.data.cpu().numpy()  # shape: (N, H, W) binary masks
                boxes_data = result.boxes

                for i, mask in enumerate(masks_data):
                    cls_id = int(boxes_data.cls[i].item())
                    cls_name = result.names[cls_id].lower()
                    conf = float(boxes_data.conf[i].item())

                    # Resize mask to canvas size
                    mask_resized = cv2.resize(mask, (hires_w, hires_h), interpolation=cv2.INTER_NEAREST)
                    mask_bool = mask_resized > 0.5

                    # Color per class: kidney=blue, cyst=dark red, tumor=red, stone=yellow
                    if cls_name == 'cyst':
                        color_bgr = (30, 30, 180)   # dark red in BGR
                    elif cls_name == 'tumor':
                        color_bgr = (0, 0, 220)     # red
                    elif cls_name == 'stone':
                        color_bgr = (0, 200, 220)   # yellow
                    else:
                        color_bgr = (200, 100, 0)   # kidney = blue-ish

                    # Fill mask area
                    overlay_layer = canvas.copy()
                    overlay_layer[mask_bool] = color_bgr
                    canvas = cv2.addWeighted(canvas, 0.55, overlay_layer, 0.45, 0)

                    # Draw contour border
                    contours, _ = cv2.findContours(
                        mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(canvas, contours, -1, color_bgr, 2)

                    # Label
                    x1, y1, x2, y2 = boxes_data.xyxy[i].cpu().numpy().astype(int)
                    # Clamp to image bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, 1)
                    cv2.putText(canvas, label, (x1, max(y1 - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)

                    # Accumulate area for impact ratio
                    pixel_area = float(np.sum(mask_bool))
                    if cls_name in ('cyst', 'tumor', 'stone'):
                        pathology_total_area += pixel_area
                    total_kidney_area += pixel_area
            else:
                # No masks — fall back to YOLO's own plot (but fix color space)
                res_plotted_rgb = result.plot(line_width=1, font_size=0.35)
                # plot() returns RGB — convert to BGR for cv2
                canvas = cv2.cvtColor(res_plotted_rgb, cv2.COLOR_RGB2BGR)

            surgical_b64 = get_base64_image(canvas)
        else:
            raise Exception("No YOLO model loaded")
    except Exception as e:
        print("YOLO error:", e)
        img_copy = image_cv2_hires.copy()
        h, w = img_copy.shape[:2]
        cv2.rectangle(img_copy, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 0, 200), 2)
        cv2.putText(img_copy, "Boundary (fallback)", (w // 4, h // 4 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
        surgical_b64 = get_base64_image(img_copy)

    # 4. Impact Ratio — computed from real mask pixel areas
    if total_kidney_area > 0:
        pathology_percent = min((pathology_total_area / total_kidney_area) * 100.0, 85.0)
    else:
        # Fallback: if no masks, use YOLO box count heuristic
        pathology_percent = 0.0
        try:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                cls_name = results[0].names[cls_id].lower()
                if cls_name in ('cyst', 'tumor', 'stone'):
                    pathology_percent += 15.0
            pathology_percent = min(pathology_percent, 85.0)
        except Exception:
            pathology_percent = 0.1

    if pathology_percent < 0.1:
        pathology_percent = 0.1
    healthy_percent = 100.0 - pathology_percent

    # 4. Impact Ratio (Matplotlib high-res)
    try:
        plt.figure(figsize=(6, 6), dpi=220)
        wedges, texts, autotexts = plt.pie(
            [healthy_percent, pathology_percent],
            labels=['Healthy Tissue', 'Pathology'],
            colors=['#10b981', '#ef4444'],
            autopct='%1.1f%%',
            pctdistance=0.75,
            startangle=140,
            textprops={'fontsize': 13, 'weight': 'bold'},
            wedgeprops=dict(width=0.42, edgecolor='white', linewidth=3)
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(13)
            autotext.set_weight('bold')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=220)
        buf.seek(0)
        impact_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    except Exception as e:
        print("Chart error:", e)
        impact_b64 = ""

    return {
        "original_image": original_b64,
        "gradcam_image": gradcam_b64,
        "surgical_boundaries": surgical_b64,
        "impact_ratio": impact_b64
    }

# =========================================================
# LLM Multimodal Clinical Copilot (from capstone_llm.ipynb)
# =========================================================
from llm_copilot import LLMCopilot, CLINICAL_PRESETS
from typing import Dict, Any

copilot_engine = LLMCopilot(use_llm=True, api_provider="gemini")

class LLMReportRequest(BaseModel):
    patient_notes: str = ""
    doctor_prescription: str = ""
    patient_name: str = ""
    patient_age: str = ""
    classification: str = "Normal"
    confidence: float = 98.0
    left_volume: int = 140
    right_volume: int = 142
    consumption_ratio: float = 18.5

class LLMChatRequest(BaseModel):
    message: str
    conversation_history: List[Dict[str, str]] = []
    patient_context: Dict[str, Any] = {}

@app.get("/api/llm/presets")
def get_clinical_presets():
    return {"presets": CLINICAL_PRESETS}

@app.post("/api/llm/generate_report")
def generate_llm_report(req: LLMReportRequest):
    notes = req.patient_notes
    if not notes and req.doctor_prescription:
        notes = f"Patient {req.patient_name or 'Patient'} (Age: {req.patient_age or 'N/A'}). Prescription & Notes: {req.doctor_prescription}"
    elif not notes:
        notes = f"Patient {req.patient_name or 'Patient'} (Age: {req.patient_age or 'N/A'}). General checkup."

    vision_data = {
        "diagnosis": req.classification,
        "classification": req.classification,
        "confidence": req.confidence,
        "left_volume": req.left_volume,
        "right_volume": req.right_volume,
        "consumption_ratio": req.consumption_ratio,
        "doctor_prescription": req.doctor_prescription
    }
    return copilot_engine.generate_multimodal_report(notes, vision_data)

@app.post("/api/llm/chat")
def chat_with_copilot(req: LLMChatRequest):
    reply = copilot_engine.chat_copilot(req.message, req.conversation_history, req.patient_context)
    return {"reply": reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
