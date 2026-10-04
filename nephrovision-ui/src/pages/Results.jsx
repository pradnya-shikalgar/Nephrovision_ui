import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Download, Printer, ArrowLeft, CheckCircle, AlertTriangle } from 'lucide-react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';

const Results = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const filename = location.state?.filename || 'Patient_CT_0042.dcm';
  const imageUrl = location.state?.imageUrl || '';
  const patientName = location.state?.patientName || '';
  const patientAge = location.state?.patientAge || '';
  const signSymptom = location.state?.signSymptom || '';
  const distance = location.state?.distance || '';
  const history = location.state?.history || '';
  const medication = location.state?.medication || '';
  const isTumor = location.state?.isTumor || false;
  const isCyst = location.state?.isCyst || false;
  const classification = location.state?.classification || "Normal Kidney Structure Detected";
  const confidence = location.state?.confidence || 98.7;
  const leftVolume = location.state?.leftVolume || 142;
  const rightVolume = location.state?.rightVolume || 138;
  const message = location.state?.message || "No visible abnormalities, cysts, or tumors identified in the highlighted regions.";
  const boundingBoxes = location.state?.boundingBoxes || [];

  return (
    <div className="page-container animate-fade-in">
      <div className="dashboard-header mb-8">
        <div>
          <Button variant="secondary" size="sm" onClick={() => navigate('/upload')} className="mb-4">
            <ArrowLeft size={16} /> New Analysis
          </Button>
          <h1>Analysis Results</h1>
          <p className="text-muted">{filename} processed successfully.</p>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <Button variant="secondary" onClick={() => navigate('/technical-details', { state: location.state })}>
            Technical Detail
          </Button>
          <Button variant="secondary" onClick={() => window.print()}>
            <Printer size={18} /> Print
          </Button>
          <Button onClick={() => navigate('/report', { state: { filename, patientName, patientAge, signSymptom, distance, history, medication, isTumor, isCyst, classification, confidence, leftVolume, rightVolume, message } })}>
            <Download size={18} /> Generate Report
          </Button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
        <Card title="Segmentation Overlay">
          <div style={{ 
            aspectRatio: '1/1', 
            backgroundColor: 'var(--color-background)', 
            borderRadius: 'var(--radius-md)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            overflow: 'hidden',
            position: 'relative'
          }}>
            {imageUrl ? (
              <img src={imageUrl} alt="CT Scan" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
            ) : (
              <p className="text-muted">No Image Available</p>
            )}
            
            {/* Real YOLO Bounding Boxes from Backend */}
            {boundingBoxes.length > 0 ? (
              boundingBoxes.map((box, index) => {
                const isTumorBox = box.class_name.toLowerCase() === 'tumor';
                const isCystBox = box.class_name.toLowerCase() === 'cyst';
                const boxColor = isTumorBox ? 'var(--color-danger)' : (isCystBox ? 'var(--color-warning)' : 'var(--color-primary)');
                const bgColor = isTumorBox ? 'rgba(239, 68, 68, 0.2)' : (isCystBox ? 'rgba(245, 158, 11, 0.2)' : 'rgba(14, 165, 233, 0.2)');
                
                return (
                  <div 
                    key={index}
                    style={{ 
                      position: 'absolute', 
                      top: `${box.y}%`, 
                      left: `${box.x}%`, 
                      width: `${box.width}%`, 
                      height: `${box.height}%`, 
                      border: `3px solid ${boxColor}`, 
                      backgroundColor: bgColor,
                      borderRadius: '8px'
                    }}
                  >
                     <span style={{ 
                       position: 'absolute', 
                       top: '-25px', 
                       left: '-3px', 
                       background: boxColor, 
                       color: 'white', 
                       padding: '2px 8px', 
                       fontSize: '12px', 
                       borderRadius: '4px',
                       fontWeight: 'bold',
                       textTransform: 'capitalize',
                       whiteSpace: 'nowrap'
                     }}>
                       {box.class_name}
                     </span>
                  </div>
                );
              })
            ) : (
              /* Fallback Simulated Overlay Bounding Box (Only if YOLO didn't return boxes but we have classification) */
              (isTumor || isCyst) && (
                <div 
                  className="animate-pulse" 
                  style={{ 
                    position: 'absolute', 
                    top: '30%', 
                    left: '45%', 
                    width: '30%', 
                    height: '40%', 
                    border: `3px solid ${isTumor ? 'var(--color-danger)' : 'var(--color-warning)'}`, 
                    backgroundColor: isTumor ? 'rgba(239, 68, 68, 0.2)' : 'rgba(245, 158, 11, 0.2)',
                    borderRadius: '10px'
                  }}
                >
                   <span style={{ 
                     position: 'absolute', 
                     top: '-25px', 
                     left: '0', 
                     background: isTumor ? 'var(--color-danger)' : 'var(--color-warning)', 
                     color: 'white', 
                     padding: '2px 6px', 
                     fontSize: '12px', 
                     borderRadius: '4px',
                     fontWeight: 'bold'
                   }}>
                     {isTumor ? 'Tumor' : 'Cyst'}
                   </span>
                </div>
              )
            )}
          </div>
        </Card>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <Card title="Findings Summary">
            {isTumor ? (
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem', marginBottom: '1.5rem' }}>
                <AlertTriangle size={24} style={{ color: 'var(--color-danger)', marginTop: '0.25rem' }} />
                <div>
                  <h4 style={{ margin: '0 0 0.5rem 0' }}>{classification}</h4>
                  <p className="text-muted" style={{ margin: 0 }}>{message}</p>
                </div>
              </div>
            ) : isCyst ? (
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem', marginBottom: '1.5rem' }}>
                <AlertTriangle size={24} style={{ color: 'var(--color-warning)', marginTop: '0.25rem' }} />
                <div>
                  <h4 style={{ margin: '0 0 0.5rem 0' }}>{classification}</h4>
                  <p className="text-muted" style={{ margin: 0 }}>{message}</p>
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem', marginBottom: '1.5rem' }}>
                <CheckCircle size={24} style={{ color: 'var(--color-success)', marginTop: '0.25rem' }} />
                <div>
                  <h4 style={{ margin: '0 0 0.5rem 0' }}>{classification}</h4>
                  <p className="text-muted" style={{ margin: 0 }}>{message}</p>
                </div>
              </div>
            )}
          </Card>
          
          <Card title="Model Metrics">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span className="text-muted">Confidence Score</span>
                  <strong>{confidence}%</strong>
                </div>
                <div style={{ width: '100%', backgroundColor: 'var(--color-border)', borderRadius: 'var(--radius-full)', height: '6px' }}>
                  <div style={{ height: '100%', backgroundColor: isTumor ? 'var(--color-danger)' : isCyst ? 'var(--color-warning)' : 'var(--color-success)', width: `${confidence}%`, borderRadius: 'var(--radius-full)' }} />
                </div>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '0.5rem', borderBottom: '1px solid var(--color-border)' }}>
                <span className="text-muted">Left Kidney Volume</span>
                <strong>{leftVolume} cm³</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '0.5rem', borderBottom: '1px solid var(--color-border)' }}>
                <span className="text-muted">Right Kidney Volume</span>
                <strong>{rightVolume} cm³</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span className="text-muted">Inference Model</span>
                <strong>PCSA KidneyNext (Augmented)</strong>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Results;
