import os
import json
import urllib.request
from typing import Dict, Any, List

# Clinical Presets matching capstone_llm.ipynb
CLINICAL_PRESETS = {
    "Case A (Severe RCC)": "Patient is a 55-year-old female presenting with gross hematuria and right-sided flank pain. History of type 2 diabetes managed with Metformin, and hypertension managed with Lisinopril. Rx: Ciprofloxacin 500mg BID for suspected UTI.",
    "Case B (Cortical Cyst)": "Patient is a 48-year-old male presenting for routine health checkup. Asymptomatic with no prior renal pathology. History of mild hyperlipidemia. Rx: Atorvastatin 20mg daily.",
    "Case C (Nephrolithiasis / Stone)": "Patient is a 39-year-old male presenting with acute, severe left-sided colicky flank pain radiating to the groin with nausea. History of dehydration and high-oxalate diet. Rx: Tamsulosin 0.4mg daily, Ibuprofen 600mg PRN.",
    "Case D (Routine Screening)": "Patient is a 32-year-old female undergoing routine abdominal screening. Healthy, normotensive, non-diabetic with no current medications."
}

DEFAULT_GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

class LLMCopilot:
    def __init__(self, use_llm: bool = True, api_provider: str = "gemini", api_key: str = None):
        self.use_llm = use_llm
        self.api_provider = api_provider
        self.api_key = api_key or DEFAULT_GEMINI_KEY
        self.model_name = "gemini-flash-latest"

    def _call_gemini_api(self, prompt: str) -> str:
        """Call Google Gemini Generative Language API directly."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.95,
                "maxOutputTokens": 2048
            }
        }
        
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode("utf-8"), 
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", "")
        return ""

    def _fallback_report(self, patient_notes: str, vision_results: Dict[str, Any]) -> str:
        """Rule-based clinical scribe fallback in case network is down."""
        diagnosis = vision_results.get("diagnosis", vision_results.get("classification", "Normal"))
        confidence = vision_results.get("confidence", 95.0)
        consumption = vision_results.get("consumption_ratio", 15.0)
        left_vol = vision_results.get("left_volume", 140)
        right_vol = vision_results.get("right_volume", 142)
        d_lower = str(diagnosis).lower()

        if "tumor" in d_lower:
            exec_syn = (
                f"• Finding: Renal mass detected ({confidence}% confidence), consistent with tumor.\n"
                f"• Organ Volumetrics: Left kidney {left_vol} cm³, Right kidney {right_vol} cm³.\n"
                f"• Tissue Involvement: Approximately {consumption}% localized parenchymal involvement.\n"
                f"• Clinical Correlation: Findings correlate with uploaded patient clinical history."
            )
            surg = (
                "• Procedure Option: Nephron-sparing partial nephrectomy or active surveillance based on lesion size.\n"
                "• Renal Function: Check serum creatinine and eGFR prior to any scheduled intervention.\n"
                "• Medication Safety: Hold Metformin and nephrotoxic medications 48 hours prior to contrast imaging.\n"
                "• Surgical Goal: Complete excision with negative margins while maximizing preserved healthy nephrons."
            )
            nutri = (
                "• Hydration Target: Drink 2.0 to 2.5 liters of neutral fluids daily to promote renal perfusion.\n"
                "• Sodium Restriction: Limit dietary sodium to < 2,000 mg/day (DASH diet) to minimize glomerular pressure.\n"
                "• Protein Control: Moderate high-quality protein (0.8 g/kg/day) to prevent hyperfiltration stress.\n"
                "• Metabolic Support: Maintain balanced blood sugar control and avoid processed nephrotoxic foods."
            )
            pat = (
                "• What Was Found: A localized spot or mass on your kidney was identified by the scan.\n"
                "• Reassurance: Modern urological treatments focus on removing only the spot while protecting the rest of your kidney.\n"
                "• Next Step: Your doctor will review the scan in detail with you and recommend a personalized plan."
            )
        elif "stone" in d_lower:
            exec_syn = (
                f"• Finding: Nephrolithiasis (kidney stone) identified with {confidence}% confidence.\n"
                f"• Organ Volumetrics: Left kidney {left_vol} cm³, Right kidney {right_vol} cm³.\n"
                f"• Tissue Integrity: Renal parenchyma is well-preserved without structural loss.\n"
                f"• Location: Stone is localized within the renal collecting system."
            )
            surg = (
                "• Management: Medical expulsive therapy for smaller stones; minimally invasive intervention if obstructing.\n"
                "• Procedure Options: Ureteroscopy (URS) or laser lithotripsy for targeted fragmentation.\n"
                "• Urgent Review: Immediate urological assessment indicated if severe colic pain or fever develops.\n"
                "• Parenchymal Protection: No tissue resection required; complete stone clearance is the primary goal."
            )
            nutri = (
                "• Hydration Target: Drink 2.5 to 3.0 liters of water daily to dilute mineral concentrations and flush deposits.\n"
                "• Sodium Control: Keep sodium below 2,000 mg/day to decrease urinary calcium excretion.\n"
                "• Dietary Balance: Moderate animal protein intake and avoid excessive high-oxalate foods (spinach, nuts, colas).\n"
                "• Citrate Intake: Add natural lemon or lime juice to water to enhance urinary citrate and prevent crystals."
            )
            pat = (
                "• What Was Found: Your scan shows a kidney stone in the urinary tract.\n"
                "• How It Heals: Many stones pass naturally with plenty of water and medicine; larger ones can be gently broken down.\n"
                "• Action Plan: Increase your daily fluid intake and follow up with your doctor for stone-specific care."
            )
        elif "cyst" in d_lower:
            exec_syn = (
                f"• Finding: Benign-appearing renal cyst identified with {confidence}% confidence.\n"
                f"• Organ Volumetrics: Symmetrical renal volumes (Left {left_vol} cm³, Right {right_vol} cm³).\n"
                f"• Architecture: Rounded, well-demarcated thin walls typical of simple cortical cysts.\n"
                f"• Status: No signs of solid invasive components or regional parenchymal distortion."
            )
            surg = (
                "• Intervention Outlook: Non-invasive routine observation; surgery is rarely necessary for simple cysts.\n"
                "• Indications for Treatment: Drainage or unroofing considered only if the cyst becomes very large or causes pain.\n"
                "• Surveillance: Standard periodic ultrasound checkup during annual health visits.\n"
                "• Renal Safety: Healthy functional parenchyma is intact without obstruction."
            )
            nutri = (
                "• Fluid Intake: Maintain 2.0 to 2.5 liters of fluids daily to support optimal renal clearance.\n"
                "• Salt Intake: Keep daily sodium under 2,000 mg to maintain healthy blood pressure.\n"
                "• Kidney Health: Standard balanced diet with fruits, vegetables, and whole grains.\n"
                "• Electrolyte Balance: Avoid high-sodium processed foods to reduce strain on renal glomeruli."
            )
            pat = (
                "• What Was Found: Your scan shows a harmless fluid pocket known as a kidney cyst.\n"
                "• Reassurance: Kidney cysts are extremely common, completely benign, and rarely cause any symptoms.\n"
                "• Next Step: No treatment or surgery is usually needed—just keep up your regular health checkups."
            )
        else:
            exec_syn = (
                f"• Finding: Normal, healthy renal architecture with {confidence}% confidence.\n"
                f"• Organ Volumetrics: Left kidney {left_vol} cm³, Right kidney {right_vol} cm³.\n"
                f"• Scan Status: Completely clear with no stones, cysts, or masses detected.\n"
                f"• Parenchyma: Smooth, uniform, and well-circumscribed bilateral kidneys."
            )
            surg = (
                "• Intervention Outlook: No surgical or procedural intervention indicated.\n"
                "• Tissue Integrity: Full bilateral anatomical preservation.\n"
                "• Follow-Up: No procedural monitoring required."
            )
            nutri = (
                "• Hydration Goal: Drink approximately 2.0 liters of water daily to maintain healthy kidney filtration.\n"
                "• Sodium Control: Maintain moderate salt intake (<2,300 mg/day) to support normal blood pressure.\n"
                "• General Wellness: Follow a balanced, nutrient-rich diet to promote long-term renal health."
            )
            pat = (
                "• What Was Found: Great news! Your kidney CT scan is completely normal and healthy.\n"
                "• Status: No evidence of stones, cysts, or abnormal masses.\n"
                "• Daily Advice: Keep staying well hydrated and active to maintain your kidney health!"
            )

        return f"""### 1. Executive Diagnostic Synthesis
{exec_syn}

### 2. Pre-Operative Surgical Risk Stratification
{surg}

### 3. Comorbidity-Aware Renal Nutrition Plan
{nutri}

### 4. Patient-Friendly Summary
{pat}"""

    @staticmethod
    def _split_report_sections(text: str) -> Dict[str, str]:
        import re
        sections = {
            "executive_synthesis": "",
            "surgical_risk_stratification": "",
            "renal_nutrition_plan": "",
            "patient_friendly_summary": ""
        }
        
        patterns = [
            ("executive_synthesis", r"(?:###\s*1\.|\b1\.\s*Executive).*?(?=(?:###\s*2\.|\b2\.\s*Pre-Operative|$))"),
            ("surgical_risk_stratification", r"(?:###\s*2\.|\b2\.\s*Pre-Operative).*?(?=(?:###\s*3\.|\b3\.\s*Comorbidity|$))"),
            ("renal_nutrition_plan", r"(?:###\s*3\.|\b3\.\s*Comorbidity).*?(?=(?:###\s*4\.|\b4\.\s*Patient|$))"),
            ("patient_friendly_summary", r"(?:###\s*4\.|\b4\.\s*Patient).*?$")
        ]
        
        for key, pat in patterns:
            match = re.search(pat, text, re.DOTALL | re.IGNORECASE)
            if match:
                sections[key] = match.group(0).strip()
                
        if not sections["executive_synthesis"] and text:
            sections["executive_synthesis"] = text
            
        return sections

    def generate_multimodal_report(self, patient_notes: str, vision_results: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the Gemini Multimodal Clinical Scribe pipeline."""
        diag = vision_results.get("diagnosis", "Normal")
        conf = vision_results.get("confidence", 98.0)
        left_vol = vision_results.get("left_volume", 140)
        right_vol = vision_results.get("right_volume", 142)
        consumption = vision_results.get("consumption_ratio", 12.0)

        prompt = f"""You are a professional Clinical Copilot for kidney health.
Write a clean, professional clinical report based on this patient's CT scan results and clinical notes.
CRITICAL FORMAT RULE: Format every section strictly in clean, simple bullet points (•). Do NOT use bold asterisks, decorative symbols, or colored highlights. Keep bullets plain, authoritative, and direct.

DATA:
- Detected Diagnosis: {diag} (Confidence: {conf}%)
- Left Kidney Volume: {left_vol} cm³, Right Kidney Volume: {right_vol} cm³
- Tissue Involvement: {consumption}%
- Patient Notes: {patient_notes}

Provide exactly these 4 sections:
### 1. Executive Diagnostic Synthesis
(3 to 4 plain bullet points: • Finding, • Organ Volumetrics, • Clinical Correlation)

### 2. Pre-Operative Surgical Risk Stratification
(3 to 4 plain bullet points: • Procedure Outlook, • Renal Function, • Medication Safety)

### 3. Comorbidity-Aware Renal Nutrition Plan
(3 to 4 plain bullet points: • Hydration Target, • Sodium Control, • Dietary Recommendations)

### 4. Patient-Friendly Summary
(3 plain bullet points: • What Was Found, • Reassurance, • Action Plan in plain language)"""

        if self.use_llm:
            try:
                print("🌐 Sending Multimodal Payload to Google Gemini API...")
                report_text = self._call_gemini_api(prompt)
                if report_text and len(report_text) > 100:
                    parsed = self._split_report_sections(report_text)
                    return {
                        "status": "success",
                        "provider": "gemini",
                        "model": self.model_name,
                        "report_text": report_text,
                        "report": parsed
                    }
                else:
                    raise Exception("Empty response from Gemini API")
            except Exception as e:
                print(f"⚠️ Gemini API Warning: {e}. Falling back to Rule-Based Medical Engine.")
                fallback_text = self._fallback_report(patient_notes, vision_results)
                parsed = self._split_report_sections(fallback_text)
                return {
                    "status": "fallback",
                    "provider": "rule_based_engine",
                    "model": "nephrovision-clinical-rules",
                    "report_text": fallback_text,
                    "report": parsed,
                    "error_detail": str(e)
                }
        else:
            fallback_text = self._fallback_report(patient_notes, vision_results)
            parsed = self._split_report_sections(fallback_text)
            return {
                "status": "success",
                "provider": "rule_based_engine",
                "model": "nephrovision-clinical-rules",
                "report_text": fallback_text,
                "report": parsed
            }

    def chat_copilot(self, user_message: str, conversation_history: List[Dict[str, str]], patient_context: Dict[str, Any]) -> str:
        """Interactive consultation with the Clinical Copilot."""
        diagnosis = patient_context.get("diagnosis", patient_context.get("classification", "Normal"))
        confidence = patient_context.get("confidence", 98.0)
        prescription = patient_context.get("doctorPrescription", patient_context.get("doctor_prescription", ""))
        patient_name = patient_context.get("patientName", patient_context.get("patient_name", "the patient"))
        left_vol = patient_context.get('left_volume', patient_context.get('leftVolume', 140))
        right_vol = patient_context.get('right_volume', patient_context.get('rightVolume', 142))
        boxes = patient_context.get('bounding_boxes', [])
        distance = patient_context.get('distance', '')
        sign_symptom = patient_context.get('sign_symptom', '')
        history_ehr = patient_context.get('history', '')
        
        # Format detected boxes/lesions
        boxes_info = []
        for b in boxes:
            cls = b.get('class_name', 'Lesion')
            bx = b.get('x', 0)
            by = b.get('y', 0)
            bw = b.get('width', 0)
            bh = b.get('height', 0)
            side = "right kidney field" if bx > 50 else "left kidney field"
            vert = "upper pole" if by < 35 else ("lower pole" if by > 65 else "mid-pole / interpolar region")
            boxes_info.append(f"{cls} detected in {side} ({vert}, coordinate approx x={bx:.1f}%, y={by:.1f}%, dimensions={bw:.1f}%x{bh:.1f}%)")

        boxes_summary = "; ".join(boxes_info) if boxes_info else "No discrete lesion bounding boxes or segmented contours reported by YOLO for this scan."

        system_instruction = f"""You are NephroVision Clinical Copilot, a friendly and supportive medical AI assistant discussing a kidney CT scan.
STYLE & FORMAT RULES:
1. Tone: Friendly, empathetic, simple, and easy to understand. Avoid complicated jargon.
2. Format: Answer in clear bullet points (•) after a short friendly opening sentence. Keep points crisp, direct, and readable.
3. Understand the exact user question before generating an answer. Answer ONLY what the user asked.
4. Base answers strictly on actual available data:
   - Finding: {diagnosis} ({confidence}% confidence)
   - Segmentation / Bounding Boxes: {boxes_summary}
   - Distance / Location Note: {distance or 'Not specifically measured'}
   - Kidney Volumes: Left {left_vol} cm³, Right {right_vol} cm³
   - Patient Info: {patient_name}, Symptoms: {sign_symptom or 'None'}, History: {history_ehr or 'None'}, Rx: {prescription or 'None'}
5. If specific measurements or coordinates are not visible on this scan, say so kindly and plainly instead of guessing."""

        history_context = ""
        for msg in conversation_history[-4:]:
            role = "User" if msg.get("role") == "user" else "Copilot"
            content = msg.get("content", msg.get("text", ""))
            history_context += f"{role}: {content}\n"

        prompt = f"{system_instruction}\n\nRecent Conversation:\n{history_context}\nUser: {user_message}\nCopilot:"
        
        # If API key is available, attempt Gemini online consultation
        if self.api_key and self.use_llm:
            try:
                res = self._call_gemini_api(prompt)
                if res and res.strip():
                    return res.strip()
            except Exception as e:
                print(f"⚠️ Gemini chat consultation warning: {e}. Utilizing NephroVision Clinical Copilot Knowledge Base.")

        # Friendly, simple, bullet-pointed answers strictly tied to user question & actual outputs
        q = user_message.lower()
        d_lower = str(diagnosis).lower()

        # 1. Location / Coordinates / Region questions
        if any(w in q for w in ["where", "location", "located", "which kidney", "position", "quadrant", "pole"]):
            if "normal" in d_lower:
                return (
                    f"Good news! No abnormal cyst, stone, or tumor was found on this scan.\n\n"
                    f"• **Left Kidney:** Normal healthy tissue ({left_vol} cm³ volume)\n"
                    f"• **Right Kidney:** Normal healthy tissue ({right_vol} cm³ volume)\n"
                    f"• **Scan Status:** Clear with no focal lesions"
                )
            
            target_boxes = [b for b in boxes if b.get('class_name', '').lower() in d_lower or b.get('class_name', '').lower() in ('tumor', 'cyst', 'stone')]
            if target_boxes:
                b = target_boxes[0]
                cls = b.get('class_name', diagnosis)
                bx = b.get('x', 0)
                by = b.get('y', 0)
                side = "Right kidney" if bx > 50 else "Left kidney"
                vert = "Upper pole (top area)" if by < 35 else ("Lower pole (bottom area)" if by > 65 else "Mid-pole (central area)")
                return (
                    f"Here is the exact location of the {cls.lower()} detected on the scan:\n\n"
                    f"• **Organ:** {side}\n"
                    f"• **Region:** {vert}\n"
                    f"• **Image Position:** Approximately x={bx:.1f}%, y={by:.1f}%\n"
                    f"{'• **Anatomical Note:** ' + distance if distance else ''}".strip()
                )
            elif distance:
                return (
                    f"Here is the location note from the scan analysis:\n\n"
                    f"• **Region:** {distance}\n"
                    f"• **Organ Side:** Right kidney field\n"
                    f"• **Details:** You can view the exact highlight in the Technical Details tab."
                )
            else:
                return (
                    f"The AI identified a {diagnosis} with {confidence}% confidence, but specific coordinate tags are not listed on this single frame:\n\n"
                    f"• **What to check:** View the **Grad-CAM heatmap** above to see the highlighted area.\n"
                    f"• **Tip:** You can check adjacent CT slices with your doctor for full 3D positioning."
                )

        # 2. Size / Dimensions / Volume / Measurements questions
        if any(w in q for w in ["size", "dimensions", "how big", "diameter", "measurement", "volume", "ratio"]):
            target_boxes = [b for b in boxes if b.get('class_name', '').lower() in d_lower or b.get('class_name', '').lower() in ('tumor', 'cyst', 'stone')]
            if target_boxes:
                b = target_boxes[0]
                cls = b.get('class_name', diagnosis)
                bw = b.get('width', 0)
                bh = b.get('height', 0)
                return (
                    f"Here are the measurement details from this scan:\n\n"
                    f"• **{cls} Dimensions:** Approximately {bw:.1f}% by {bh:.1f}% of the scan field\n"
                    f"• **Left Kidney Volume:** {left_vol} cm³\n"
                    f"• **Right Kidney Volume:** {right_vol} cm³\n"
                    f"• **Status:** Localized without significant kidney enlargement"
                )
            elif "normal" in d_lower:
                return (
                    f"Both kidneys show healthy, standard dimensions:\n\n"
                    f"• **Left Kidney Volume:** {left_vol} cm³\n"
                    f"• **Right Kidney Volume:** {right_vol} cm³\n"
                    f"• **Tissue:** 100% healthy renal parenchyma with no mass"
                )
            else:
                return (
                    f"Here are the recorded volume measurements:\n\n"
                    f"• **Left Kidney:** {left_vol} cm³\n"
                    f"• **Right Kidney:** {right_vol} cm³\n"
                    f"• **Millimeter Dimensions:** Not directly calibrated on this single slice; clinical caliper review is advised."
                )

        # 3. Boundaries / Contours / Margin questions on the image
        if any(w in q for w in ["boundary", "boundaries", "contour", "outline", "shape", "edges"]):
            if "normal" in d_lower:
                return (
                    "The kidney boundaries look smooth and healthy:\n\n"
                    "• **Shape:** Well-circumscribed and natural bean-shaped outline\n"
                    "• **Parenchyma:** Uniform, even texture throughout\n"
                    "• **Borders:** Crisp with no irregular bulges"
                )
            if "tumor" in d_lower:
                return (
                    "Here is what the image boundaries show:\n\n"
                    "• **Contour:** Slightly irregular border in the affected kidney zone\n"
                    "• **Tissue Contrast:** Clearly distinguished from normal surrounding parenchyma\n"
                    "• **Visual Aid:** Open **Technical Details** to see the colored surgical boundary outline."
                )
            if "cyst" in d_lower:
                return (
                    "The cyst shows typical benign characteristics:\n\n"
                    "• **Shape:** Smooth, round, and thin-walled\n"
                    "• **Edges:** Sharp and well-demarcated from the kidney tissue\n"
                    "• **Interior:** Low-density, clear fluid-like appearance"
                )
            if "stone" in d_lower:
                return (
                    "The stone has distinct, high-contrast features:\n\n"
                    "• **Boundary:** Dense, bright focal mineral calcification\n"
                    "• **Outline:** Outlined clearly by the dual-tone marker ring on the image\n"
                    "• **Location:** Contained within the urinary collecting tract"
                )

        # 4. Grad-CAM / Heatmap / AI Attention questions
        if any(w in q for w in ["gradcam", "grad-cam", "heatmap", "attention", "highlight", "why is", "what is highlighted"]):
            if "normal" in d_lower:
                return (
                    "For a normal scan, Grad-CAM keeps the scan clean and natural:\n\n"
                    "• **Why clean?** No abnormal focal spots were detected to create a red hotspot.\n"
                    "• **What it means:** The AI verified that the kidney tissue is consistently healthy."
                )
            return (
                f"The Grad-CAM heatmap shows you exactly where the AI focused:\n\n"
                f"• **Red & Yellow Areas:** The most important spot that led the model to detect {diagnosis}\n"
                f"• **Confidence:** {confidence}% certainty from the neural network\n"
                f"• **Surrounding Area:** Stays natural CT gray so you can easily see the anatomy"
            )

        # 5. Classification / Prediction / Confidence questions
        if any(w in q for w in ["classification", "confidence", "prediction", "accuracy", "diagnosis", "what did the model find", "what is this"]):
            return (
                f"Here is a summary of the AI analysis for this scan:\n\n"
                f"• **Prediction:** {diagnosis}\n"
                f"• **Confidence Level:** {confidence}%\n"
                f"• **Model:** PCSA-KidneyNeXt dual-stage architecture with YOLO verification\n"
                f"• **Next Step:** Review with your urologist for confirmation"
            )

        # 6. Surgical Margins questions (targeted)
        if "surgical margin" in q or ("margin" in q and ("surgery" in q or "resect" in q or "recommend" in q)):
            if "tumor" in d_lower:
                return (
                    "Here are the urological recommendations for surgical margins:\n\n"
                    "• **Target Margin:** 1 mm to 5 mm of normal tissue is standard for partial nephrectomy\n"
                    "• **Enucleation Option:** Removing the tumor right along its natural shell saves maximum healthy kidney\n"
                    "• **Goal:** Complete cancer removal while protecting long-term kidney function"
                )
            elif "stone" in d_lower:
                return (
                    "Surgical margins do not apply to stones:\n\n"
                    "• **Primary Goal:** Break up and completely clear the stone\n"
                    "• **Approach:** Gentle lithotripsy or endoscopic extraction"
                )
            elif "cyst" in d_lower:
                return (
                    "Margins are not needed for simple cysts:\n\n"
                    "• **Nature:** Cysts are benign fluid pockets\n"
                    "• **Treatment:** If symptomatic, the cyst roof is simply unroofed or drained"
                )
            else:
                return (
                    "No surgical margins are needed:\n\n"
                    "• Both kidneys are completely normal and healthy."
                )

        # 7. Follow-up interval questions
        if any(w in q for w in ["interval", "follow-up", "follow up", "when should", "next scan"]):
            if "tumor" in d_lower:
                return (
                    "Here is the standard imaging schedule:\n\n"
                    "• **Post-Treatment:** CT or MRI at 3 to 6 months, then yearly for 5 years\n"
                    "• **Active Observation:** Repeat scans every 3 to 6 months for the first year to monitor size"
                )
            elif "stone" in d_lower:
                return (
                    "Recommended checkup schedule for stones:\n\n"
                    "• **Post-Procedure:** Ultrasound or low-dose X-ray in 4 to 6 weeks to confirm it's gone\n"
                    "• **Prevention:** 24-hour urine analysis in 6 months to prevent new stones"
                )
            elif "cyst" in d_lower:
                return (
                    "Follow-up for simple cysts:\n\n"
                    "• **Routine:** Usually no repeat CT needed unless pain occurs\n"
                    "• **Checkup:** Simple yearly ultrasound during regular wellness exams"
                )
            else:
                return (
                    "No follow-up CT is necessary:\n\n"
                    "• **Next Scan:** Only needed if new urinary or flank symptoms develop\n"
                    "• **Routine:** Standard annual health checkups"
                )

        # 8. Diabetes / Medication risk questions
        if "diabetes" in q or "metformin" in q:
            return (
                "Important guidance for diabetic patients having kidney scans or procedures:\n\n"
                "• **Metformin Safety:** Stop Metformin 48 hours before contrast imaging or surgery to avoid complications\n"
                "• **Blood Sugar:** Keep blood glucose steady (ideally 140–180 mg/dL perioperatively)\n"
                "• **Kidney Protection:** Drink plenty of fluids and re-test kidney blood levels (eGFR) before restarting meds"
            )

        # 9. Patient summary questions
        if any(w in q for w in ["explain", "patient", "simply", "plain language", "tell the patient"]):
            if "tumor" in d_lower:
                return (
                    f"Here is a simple, reassuring summary for {patient_name}:\n\n"
                    f"• **Finding:** The scan detected a localized mass on the kidney ({confidence}% confidence).\n"
                    f"• **Good News:** Modern procedures focus on removing only the affected spot while keeping the rest of the kidney healthy.\n"
                    f"• **Action Plan:** A urologist will discuss the scan with you and personalize a straightforward care plan."
                )
            elif "stone" in d_lower:
                return (
                    f"Here is a simple summary for {patient_name}:\n\n"
                    f"• **Finding:** The scan shows a kidney stone ({confidence}% confidence).\n"
                    f"• **What it is:** A harmless mineral deposit that formed in the kidney.\n"
                    f"• **Action Plan:** Many pass naturally with extra water and mild medicine; larger ones can be gently broken up."
                )
            elif "cyst" in d_lower:
                return (
                    f"Here is a simple summary for {patient_name}:\n\n"
                    f"• **Finding:** The scan shows a benign kidney cyst ({confidence}% confidence).\n"
                    f"• **What it is:** A very common, harmless fluid pocket.\n"
                    f"• **Action Plan:** In almost all cases, no surgery or medicine is needed—just an occasional routine check."
                )
            else:
                return (
                    f"Great news for {patient_name}:\n\n"
                    f"• **Finding:** Your kidney CT scan is completely normal ({confidence}% confidence).\n"
                    f"• **Health:** No stones, cysts, or masses were detected.\n"
                    f"• **Tip:** Keep drinking plenty of water and staying active!"
                )

        # Direct context-grounded fallback answering the user
        return (
            f"Here are the key takeaways for {patient_name}:\n\n"
            f"• **Diagnosis:** {diagnosis} ({confidence}% confidence)\n"
            f"• **Left Kidney Volume:** {left_vol} cm³\n"
            f"• **Right Kidney Volume:** {right_vol} cm³\n"
            f"{'• **Location Note:** ' + distance if distance else ''}\n"
            f"• **Need More Details?** Feel free to ask about location, size, boundaries, or next steps!"
        ).strip()
