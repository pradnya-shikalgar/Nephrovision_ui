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

        return f"""### 1. Executive Diagnostic Synthesis
The patient's CT scan indicates a **{diagnosis}** with a confidence score of {confidence}%.
- Organ Volumetrics: Left kidney {left_vol} cm³, Right kidney {right_vol} cm³.
- Clinical Context Ingested: {patient_notes}

### 2. Pre-Operative Surgical Risk Stratification
**Protocol Recommendation:** {'Nephron-sparing partial resection indicated' if 'tumor' in diagnosis.lower() else 'Routine non-invasive surveillance'}
- **Tissue Involvement:** Lesion mass corresponds to approximately {consumption}% of regional parenchyma.
- **Risk Evaluation:** Review comorbid conditions (hypertension, diabetes) prior to elective interventions. Hold nephrotoxic medications and Metformin before contrast imaging.

### 3. Comorbidity-Aware Renal Nutrition Plan
- **Electrolyte & Sodium Control:** Restrict dietary sodium to < 2000mg/day (DASH framework) to protect functional glomeruli.
- **Hydration Protocol:** Maintain target intake of 2.0L - 2.5L daily to flush metabolites.
- **Protein Intake:** Controlled high-quality protein (0.8g/kg/day) to prevent hyperfiltration.

### 4. Patient-Friendly Summary
Our AI system identified a {diagnosis.lower()} on your kidney scan. Your medical team will review the detailed surgical and dietary recommendations to provide personalized, comprehensive care."""

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

        prompt = f"""You are an advanced AI Clinical Copilot and Multimodal Scribe for Nephrology and Urological Oncology.
Synthesize the following patient EHR notes and Vision Diagnostic findings from a renal CT scan into a 4-section clinical report.

SECTIONS REQUIRED:
1. Executive Diagnostic Synthesis
2. Pre-Operative Surgical Risk Stratification
3. Comorbidity-Aware Renal Nutrition Plan
4. Patient-Friendly Summary

PATIENT CLINICAL RECORD:
{patient_notes}

CT SCAN AUTOMATED VISION FINDINGS:
- Diagnosis / Detected Abnormality: {diag}
- Confidence: {conf}%
- Left Kidney Volume: {left_vol} cm³
- Right Kidney Volume: {right_vol} cm³
- Surgical Consumption / Pathology Ratio: {consumption}%
- Deep Learning Architecture: PCSA KidneyNeXt + YOLOv8 Specialist

INSTRUCTIONS:
1. In Section 1, synthesize image findings, volumetric discrepancies, and clinical presentation.
2. In Section 2, provide actionable surgical protocol choices (e.g. Partial Nephrectomy vs Surveillance), assess perioperative medication holds (e.g. Metformin, ACE-inhibitors), and vascular risks.
3. In Section 3, specify sodium limits (<2000mg/day), protein targets (0.8g/kg/day), glycemic guidelines, and hydration targets tailored to the specific diagnosis.
4. In Section 4, explain the results compassionately and clearly for the patient in plain English.
Keep tone authoritative, medically accurate, and structured with bullet points."""

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
        """Interactive consultation with the Gemini Clinical Copilot."""
        diagnosis = patient_context.get("classification", "Normal")
        confidence = patient_context.get("confidence", 98.0)
        prescription = patient_context.get("doctorPrescription", "")
        patient_name = patient_context.get("patientName", "Patient")
        
        system_instruction = f"""You are the NephroVision AI Clinical Copilot, an expert medical AI assistant specializing in nephrology and urological surgery.
Current Patient Case:
- Name: {patient_name}
- Diagnosis: {diagnosis} (Confidence {confidence}%)
- Doctor Prescription / Notes: {prescription}
- Left Volume: {patient_context.get('leftVolume', 140)} cm³, Right Volume: {patient_context.get('rightVolume', 142)} cm³

Answer the doctor or patient's question clearly, clinically, and concisely. If discussing medications or surgery, emphasize evidence-based urology guidelines (AUA/EAU)."""

        history_context = ""
        for msg in conversation_history[-4:]:
            role = "User" if msg.get("role") == "user" else "Copilot"
            history_context += f"{role}: {msg.get('text', '')}\n"

        prompt = f"{system_instruction}\n\nRecent Conversation:\n{history_context}\nUser: {user_message}\nCopilot:"
        
        try:
            return self._call_gemini_api(prompt)
        except Exception as e:
            return f"The NephroVision AI Copilot is currently operating in offline mode: Based on the {diagnosis} diagnosis and clinical notes, standard guidelines recommend reviewing renal panel tests and urological consultation. ({e})"
