import React, { useEffect, useState, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Loader2, AlertCircle, RefreshCw, PlayCircle, ArrowLeft } from 'lucide-react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';

const Analyze = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const file = location.state?.file;
  const imageUrl = location.state?.imageUrl;
  const patientName = location.state?.patientName || '';
  const patientAge = location.state?.patientAge || '';
  const doctorPrescription = location.state?.doctorPrescription || '';
  const filename = location.state?.filename || 'Patient_CT_0042.dcm';
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);

  const completeAnalysis = useCallback((data) => {
    setProgress(100);
    setTimeout(() => {
      const analysisResult = { 
        filename, 
        imageUrl,
        patientName,
        patientAge,
        doctorPrescription: data.doctor_prescription || doctorPrescription || '',
        signSymptom: data.sign_symptom || 'None reported',
        distance: data.distance || 'N/A',
        history: data.history || 'No prior medical history',
        medication: data.medication || 'None',
        isTumor: Boolean(data.is_tumor),
        isCyst: Boolean(data.is_cyst),
        classification: data.classification || 'Normal Kidney Structure',
        confidence: data.confidence || 98.5,
        leftVolume: data.left_volume || 140,
        rightVolume: data.right_volume || 142,
        message: data.message || 'No abnormal findings detected.',
        boundingBoxes: data.bounding_boxes || [],
        date: new Date().toLocaleString()
      };

      const existingHistory = JSON.parse(localStorage.getItem('recentAnalyses') || '[]');
      localStorage.setItem('recentAnalyses', JSON.stringify([analysisResult, ...existingHistory].slice(0, 10)));

      navigate('/results', { state: { ...analysisResult, file } });
    }, 500);
  }, [filename, imageUrl, patientName, patientAge, doctorPrescription, navigate, file]);

  const runSimulatedInference = useCallback(() => {
    setError(null);
    setProgress(20);

    const interval = setInterval(() => {
      setProgress(p => (p >= 90 ? 90 : p + 20));
    }, 300);

    setTimeout(() => {
      clearInterval(interval);
      const filenameLower = filename.toLowerCase();
      let classification = "Normal Kidney Structure";
      let isTumor = false;
      let isCyst = false;
      let message = "No visible abnormalities, cysts, or tumors identified in the highlighted regions.";
      let signSymptom = "None (Routine checkup)";
      let distance = "N/A";
      let history = "Healthy";
      let medication = "None";
      let rightVol = 138;
      let confidence = 98.4;

      if (filenameLower.includes('tumor') || filenameLower.includes('abnormal') || filenameLower.includes('tumer')) {
        classification = "Tumor Detected";
        isTumor = true;
        message = "An abnormal mass/tumor has been identified in the right kidney region. Immediate clinical review is recommended.";
        signSymptom = "Hematuria, Flank Pain, Palpable Mass";
        distance = "4.2 cm from renal pelvis";
        history = "Smoker, Hypertension";
        medication = "Amlodipine 5mg";
        rightVol = 154;
        confidence = 99.2;
      } else if (filenameLower.includes('cyst')) {
        classification = "Cyst Detected";
        isCyst = true;
        message = "A benign-appearing cyst was identified in the renal cortex. Routine monitoring is advised.";
        signSymptom = "Asymptomatic (Incidental finding)";
        distance = "Cortical surface";
        history = "No relevant history";
        medication = "None";
        rightVol = 146;
        confidence = 97.8;
      }

      completeAnalysis({
        classification,
        confidence,
        is_tumor: isTumor,
        is_cyst: isCyst,
        left_volume: 140,
        right_volume: rightVol,
        message,
        sign_symptom: signSymptom,
        distance,
        history,
        medication,
        doctor_prescription: doctorPrescription,
        bounding_boxes: isTumor ? [
          { class_name: 'Tumor', x: 42, y: 35, width: 28, height: 32 }
        ] : isCyst ? [
          { class_name: 'Cyst', x: 45, y: 38, width: 22, height: 24 }
        ] : []
      });
    }, 1200);
  }, [completeAnalysis, filename, doctorPrescription]);

  const performAnalysis = useCallback(async () => {
    if (!file) {
      navigate('/upload');
      return;
    }

    setError(null);
    setProgress(10);

    const interval = setInterval(() => {
      setProgress(p => (p >= 90 ? 90 : p + Math.random() * 15));
    }, 400);

    try {
      const formData = new FormData();
      formData.append('file', file);
      if (doctorPrescription) formData.append('doctor_prescription', doctorPrescription);
      if (patientName) formData.append('patient_name', patientName);
      if (patientAge) formData.append('patient_age', patientAge);

      let response;
      try {
        response = await fetch('http://127.0.0.1:8000/api/analyze', {
          method: 'POST',
          body: formData
        });
      } catch {
        response = await fetch('http://localhost:8000/api/analyze', {
          method: 'POST',
          body: formData
        });
      }

      if (!response.ok) throw new Error("API Request Failed with status: " + response.status);
      const data = await response.json();
      
      clearInterval(interval);
      completeAnalysis(data);
    } catch (err) {
      console.error(err);
      clearInterval(interval);
      setError("Analysis failed to connect to backend server. Make sure FastAPI is running on port 8000.");
    }
  }, [file, doctorPrescription, patientName, patientAge, navigate, completeAnalysis]);

  useEffect(() => {
    performAnalysis();
  }, [performAnalysis]);

  return (
    <div className="page-container animate-fade-in" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '80vh' }}>
      <Card className="text-center" style={{ maxWidth: '540px', width: '100%', padding: '2.5rem' }}>
        {error ? (
          <div>
            <div style={{ 
              width: '56px', 
              height: '56px', 
              borderRadius: '50%', 
              backgroundColor: 'var(--color-danger-bg)', 
              color: 'var(--color-danger)', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center', 
              margin: '0 auto 1.25rem' 
            }}>
              <AlertCircle size={32} />
            </div>
            <h2 className="mb-2" style={{ color: 'var(--color-danger)' }}>Connection Error</h2>
            <p className="text-muted mb-4" style={{ fontSize: '0.95rem', lineHeight: '1.5' }}>
              {error}
            </p>
            <div style={{ backgroundColor: 'var(--color-background)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-md)', padding: '1rem', marginBottom: '1.5rem', textAlign: 'left', fontSize: '0.85rem' }}>
              <strong>Troubleshooting Guide:</strong>
              <ul style={{ paddingLeft: '1.25rem', marginTop: '0.5rem', color: 'var(--color-text-muted)' }}>
                <li>Make sure FastAPI server is active: <code>./backend/run_backend.sh</code></li>
                <li>Or click "Simulated Mode" below to proceed offline immediately.</li>
              </ul>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center' }}>
                <Button variant="secondary" onClick={() => navigate('/upload')}>
                  <ArrowLeft size={16} /> Go Back
                </Button>
                <Button onClick={performAnalysis}>
                  <RefreshCw size={16} /> Retry Connection
                </Button>
              </div>
              <Button variant="secondary" onClick={runSimulatedInference} style={{ borderStyle: 'dashed' }}>
                <PlayCircle size={16} /> Run in Demo / Simulated Mode (Offline)
              </Button>
            </div>
          </div>
        ) : (
          <div>
            <Loader2 className="animate-spin" size={60} style={{ color: 'var(--color-primary)', margin: '0 auto 1.5rem' }} />
            <h2 className="mb-2">Analyzing CT Scan...</h2>
            <p className="text-muted mb-6">Running NephroVision PCSA Deep Learning Model.</p>
        
            <div style={{ width: '100%', backgroundColor: 'var(--color-border)', borderRadius: 'var(--radius-full)', height: '8px', overflow: 'hidden' }}>
              <div 
                style={{ 
                  height: '100%', 
                  backgroundColor: 'var(--color-primary)', 
                  width: `${Math.min(progress, 100)}%`,
                  transition: 'width 0.3s ease-out'
                }} 
              />
            </div>
            <p className="mt-4 text-muted" style={{ fontSize: '0.875rem' }}>{Math.min(Math.round(progress), 100)}% Complete</p>
          </div>
        )}
      </Card>
    </div>
  );
};

export default Analyze;
