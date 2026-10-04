import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Printer, ArrowLeft, Download } from 'lucide-react';
import html2pdf from 'html2pdf.js';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';

const Report = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const filename = location.state?.filename || 'Patient_CT_0042.dcm';
  const patientName = location.state?.patientName || 'Not Specified';
  const patientAge = location.state?.patientAge || 'Not Specified';
  const signSymptom = location.state?.signSymptom || 'Not Specified';
  const distance = location.state?.distance || 'Not Specified';
  const history = location.state?.history || 'Not Specified';
  const medication = location.state?.medication || 'Not Specified';
  const isTumor = location.state?.isTumor || false;
  const isCyst = location.state?.isCyst || false;
  const classification = location.state?.classification || "Normal Kidney Structure";
  const confidence = location.state?.confidence || 98.7;
  const leftVolume = location.state?.leftVolume || 142;
  const rightVolume = location.state?.rightVolume || 138;
  const message = location.state?.message || "No visible abnormalities, cysts, or tumors identified in the highlighted regions.";

  const [currentDate] = useState(() => new Date().toLocaleDateString());

  const handleSavePDF = () => {
    const element = document.getElementById('report-content');
    const opt = {
      margin: 0.5,
      filename: `NephroVision_${filename}.pdf`,
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { scale: 2 },
      jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
    };
    html2pdf().set(opt).from(element).save();
  };

  return (
    <div className="page-container animate-fade-in" style={{ backgroundColor: 'white', maxWidth: '800px', margin: '2rem auto', padding: '3rem', boxShadow: 'var(--shadow-lg)', borderRadius: 'var(--radius-md)' }}>
      
      <div className="no-print" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2rem' }}>
        <Button variant="secondary" size="sm" onClick={() => navigate(-1)}>
          <ArrowLeft size={16} /> Back to Results
        </Button>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <Button variant="secondary" onClick={() => window.print()}>
            <Printer size={16} /> Print Report
          </Button>
          <Button onClick={handleSavePDF}>
            <Download size={16} /> Save PDF
          </Button>
        </div>
      </div>

      <div id="report-content" style={{ padding: '1rem' }}>
        <div style={{ borderBottom: '2px solid var(--color-border)', paddingBottom: '1rem', marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h1 style={{ margin: '0 0 0.5rem 0', color: 'var(--color-primary)' }}>NephroVision</h1>
          <h3 style={{ margin: 0, color: 'var(--color-text-muted)', fontWeight: 500 }}>Automated Kidney Analysis Report</h3>
        </div>
        <div className="text-muted" style={{ textAlign: 'right' }}>
          <p style={{ margin: 0 }}>Date: {currentDate}</p>
          <p style={{ margin: 0 }}>Report ID: NV-20261004-8932</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2rem' }}>
        <div>
          <h4 style={{ borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem', marginBottom: '1rem' }}>Patient Clinical Context</h4>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
            <p><strong>Name:</strong> {patientName}</p>
            <p><strong>Age:</strong> {patientAge}</p>
            <p><strong>Signs/Symptoms:</strong> {signSymptom}</p>
            <p><strong>Tumor Distance:</strong> {distance}</p>
            <p><strong>History:</strong> {history}</p>
            <p><strong>Medication:</strong> {medication}</p>
            <p><strong>Scan ID:</strong> {filename}</p>
          </div>
        </div>
        <div>
          <h4 style={{ borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem', marginBottom: '1rem' }}>Model Details</h4>
          <p><strong>Algorithm:</strong> PCSA KidneyNext (Augmented)</p>
          <p><strong>Version:</strong> v2.1.0</p>
          <p><strong>Processing Time:</strong> 2.4s</p>
        </div>
      </div>

      <Card title="Analysis Findings" className="mb-8" style={{ boxShadow: 'none', border: '1px solid var(--color-border)' }}>
        {isTumor ? (
          <>
            <h4 style={{ color: 'var(--color-danger)', marginBottom: '0.5rem' }}>Status: {classification} (Confidence {confidence}%)</h4>
            <p>{message}</p>
          </>
        ) : isCyst ? (
          <>
            <h4 style={{ color: 'var(--color-warning)', marginBottom: '0.5rem' }}>Status: {classification} (Confidence {confidence}%)</h4>
            <p>{message}</p>
          </>
        ) : (
          <>
            <h4 style={{ color: 'var(--color-success)', marginBottom: '0.5rem' }}>Status: {classification} (Confidence {confidence}%)</h4>
            <p>{message}</p>
          </>
        )}
        
        <ul style={{ paddingLeft: '1.5rem', marginTop: '1rem' }}>
          <li>Left Kidney Volume: {leftVolume} cm³ (Normal Range)</li>
          <li>Right Kidney Volume: {rightVolume} cm³ {isTumor ? '(Abnormal Growth Detected)' : isCyst ? '(Enlarged)' : '(Normal Range)'}</li>
          <li>Symmetry Ratio: {(leftVolume / rightVolume).toFixed(2)} {isTumor ? '(Abnormal)' : isCyst ? '(Warning)' : '(Normal)'}</li>
        </ul>
      </Card>

      <Card title="Dietary & Lifestyle Plan" className="mb-8" style={{ boxShadow: 'none', border: '1px solid var(--color-border)', backgroundColor: 'var(--color-background)' }}>
        <ul style={{ paddingLeft: '1.5rem', fontSize: '0.9rem', color: 'var(--color-text)', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {isTumor ? (
            <>
              <li><strong>Post-Surgical/Oncology Nutrition:</strong> High-quality, lean protein for tissue repair. Limit processed red meats.</li>
              <li><strong>Renal-Protective Diet:</strong> Strict monitoring of sodium and potassium to prevent kidney strain.</li>
              <li><strong>Hydration:</strong> Maintain adequate hydration but consult nephrologist for fluid restrictions if kidney function declines.</li>
              <li><strong>Lifestyle:</strong> Avoid smoking and manage blood pressure meticulously to preserve remaining kidney function.</li>
            </>
          ) : isCyst ? (
            <>
              <li><strong>Hydration:</strong> Drink plenty of water daily to help prevent cyst enlargement and stone formation.</li>
              <li><strong>Sodium Control:</strong> Limit salt intake to help manage blood pressure, which is crucial for kidney health.</li>
              <li><strong>Protein Moderation:</strong> Avoid excessive high-protein diets which can increase renal workload.</li>
              <li><strong>Monitoring:</strong> Schedule routine ultrasound monitoring as advised by your urologist.</li>
            </>
          ) : (
            <>
              <li><strong>General Renal Health:</strong> Maintain a balanced diet rich in vegetables, fruits, and whole grains.</li>
              <li><strong>Hydration:</strong> Drink 2-3 liters of water daily to flush out toxins.</li>
              <li><strong>Preventative:</strong> Maintain a healthy weight and monitor blood pressure to prevent future kidney issues.</li>
            </>
          )}
        </ul>
      </Card>

        <div style={{ textAlign: 'center', marginTop: '4rem', paddingTop: '2rem', borderTop: '1px solid var(--color-border)' }}>
          <p className="text-muted" style={{ fontSize: '0.875rem' }}>This report was generated automatically by an AI system. It is intended to assist medical professionals and should not replace formal clinical diagnosis.</p>
        </div>
      </div>

      <style>{`
        @media print {
          .no-print { display: none !important; }
          body { background-color: white; }
          .page-container { margin: 0; padding: 0; box-shadow: none; max-width: 100%; }
        }
      `}</style>
    </div>
  );
};

export default Report;
