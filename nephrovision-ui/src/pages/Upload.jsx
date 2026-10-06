import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, ArrowLeft, User, FileText, Sparkles } from 'lucide-react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import FileUpload from '../components/ui/FileUpload';

const CLINICAL_PRESETS = [
  { label: 'Custom Case / Enter Manually', name: '', age: '', rx: '' },
  { 
    label: 'Case A: Severe RCC (Tumor + Diabetes/HTN)', 
    name: 'Sarah Jenkins', 
    age: '55', 
    rx: 'Patient is a 55-year-old female presenting with gross hematuria and right-sided flank pain. History of type 2 diabetes managed with Metformin, and hypertension managed with Lisinopril. Rx: Ciprofloxacin 500mg BID for suspected UTI.' 
  },
  { 
    label: 'Case B: Cortical Cyst (Mild Hyperlipidemia)', 
    name: 'David Miller', 
    age: '48', 
    rx: 'Patient is a 48-year-old male presenting for routine health checkup. Asymptomatic with no prior renal pathology. History of mild hyperlipidemia. Rx: Atorvastatin 20mg daily.' 
  },
  { 
    label: 'Case C: Nephrolithiasis (Acute Kidney Stone)', 
    name: 'Robert Chen', 
    age: '39', 
    rx: 'Patient is a 39-year-old male presenting with acute, severe left-sided colicky flank pain radiating to groin with nausea. History of dehydration and high-oxalate diet. Rx: Tamsulosin 0.4mg daily, Ibuprofen 600mg PRN.' 
  }
];

const Upload = () => {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [patientName, setPatientName] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const [doctorPrescription, setDoctorPrescription] = useState('');

  const handleSelectPreset = (presetLabel) => {
    const found = CLINICAL_PRESETS.find(p => p.label === presetLabel);
    if (found) {
      setPatientName(found.name);
      setPatientAge(found.age);
      setDoctorPrescription(found.rx);
    }
  };

  const handleFileSelect = (file) => {
    setSelectedFile(file);
  };

  const handleAnalyze = () => {
    if (selectedFile) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const imageUrl = reader.result;
        navigate('/analyze', { 
          state: { 
            file: selectedFile, 
            filename: selectedFile.name, 
            imageUrl, 
            patientName, 
            patientAge,
            doctorPrescription 
          } 
        });
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  return (
    <div className="page-container animate-fade-in">
      <Button variant="secondary" size="sm" onClick={() => navigate(-1)} className="mb-4">
        <ArrowLeft size={16} /> Back
      </Button>
      
      <div className="text-center mb-8">
        <h1 className="mb-4">Upload CT Scan Image</h1>
        <p className="text-muted">Supported formats: DICOM (.dcm), PNG, JPEG. Max size: 50MB.</p>
      </div>

      <div style={{ maxWidth: '600px', margin: '0 auto' }}>
        <Card>
          <FileUpload onFileSelect={handleFileSelect} />
          
          {selectedFile && (
            <div style={{ marginTop: '2rem', display: 'flex', flexDirection: 'column', gap: '1rem', textAlign: 'left' }}>
              <div style={{ backgroundColor: 'rgba(59, 130, 246, 0.08)', border: '1px solid rgba(59, 130, 246, 0.25)', borderRadius: 'var(--radius-md)', padding: '0.875rem 1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: '#60a5fa', fontWeight: 600, fontSize: '0.875rem' }}>
                  <Sparkles size={16} /> Clinical LLM Copilot Presets (Fast Testing)
                </div>
                <select
                  onChange={(e) => handleSelectPreset(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.5rem 0.75rem',
                    borderRadius: 'var(--radius-sm)',
                    backgroundColor: 'var(--color-background)',
                    border: '1px solid var(--color-border)',
                    color: 'var(--color-text)',
                    fontSize: '0.875rem',
                    outline: 'none',
                    cursor: 'pointer'
                  }}
                  defaultValue=""
                >
                  <option value="" disabled>Select a pre-configured clinical case or enter below...</option>
                  {CLINICAL_PRESETS.map((p, idx) => (
                    <option key={idx} value={p.label}>{p.label}</option>
                  ))}
                </select>
                <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '0.375rem' }}>
                  Autofills patient clinical history & prescription notes designed for the Gemini Multimodal Copilot.
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500, fontSize: '0.875rem' }}>Patient Name (Optional)</label>
                  <div style={{ display: 'flex', alignItems: 'center', backgroundColor: 'var(--color-background)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-md)', padding: '0.75rem 1.25rem' }}>
                    <User size={20} className="text-muted" style={{ marginRight: '0.75rem' }} />
                    <input 
                      type="text" 
                      placeholder="Enter full name" 
                      value={patientName}
                      onChange={(e) => setPatientName(e.target.value)}
                      style={{ border: 'none', background: 'transparent', outline: 'none', width: '100%', color: 'var(--color-text)', fontSize: '1rem' }}
                    />
                  </div>
                </div>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500, fontSize: '0.875rem' }}>Patient Age (Optional)</label>
                  <div style={{ display: 'flex', alignItems: 'center', backgroundColor: 'var(--color-background)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-md)', padding: '0.75rem 1.25rem' }}>
                    <input 
                      type="number" 
                      placeholder="Enter age" 
                      value={patientAge}
                      onChange={(e) => setPatientAge(e.target.value)}
                      style={{ border: 'none', background: 'transparent', outline: 'none', width: '100%', color: 'var(--color-text)', fontSize: '1rem' }}
                    />
                  </div>
                </div>
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500, fontSize: '0.875rem' }}>Doctor Prescription (Optional)</label>
                <div style={{ display: 'flex', alignItems: 'flex-start', backgroundColor: 'var(--color-background)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-md)', padding: '0.75rem 1.25rem' }}>
                  <FileText size={20} className="text-muted" style={{ marginRight: '0.75rem', marginTop: '0.25rem', flexShrink: 0 }} />
                  <textarea 
                    placeholder="Enter doctor's prescription, notes, or clinical medication instructions..." 
                    value={doctorPrescription}
                    onChange={(e) => setDoctorPrescription(e.target.value)}
                    rows={3}
                    style={{ border: 'none', background: 'transparent', outline: 'none', width: '100%', color: 'var(--color-text)', fontSize: '0.95rem', resize: 'vertical', fontFamily: 'inherit' }}
                  />
                </div>
              </div>
            </div>
          )}

          <div style={{ marginTop: '2rem', display: 'flex', justifyContent: 'flex-end' }}>
            <Button 
              size="lg" 
              disabled={!selectedFile}
              onClick={handleAnalyze}
            >
              Start Analysis <ArrowRight size={18} />
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Upload;
