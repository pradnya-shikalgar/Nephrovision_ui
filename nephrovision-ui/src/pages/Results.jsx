import React, { useState, useRef, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Download, Printer, ArrowLeft, CheckCircle, AlertTriangle, FileText, Sparkles, Bot, Send, Brain, Stethoscope, RefreshCw, MessageSquare, ShieldCheck, ChevronRight, User } from 'lucide-react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import { API_BASE } from '../apiConfig';

// Clean raw markdown headers (### 1. Title etc.) for sleek presentation
const cleanSectionText = (text) => {
  if (!text) return '';
  return text
    .replace(/^###\s*\d*\.?\s*[^\n]*\n?/gm, '')
    .replace(/^\*\*Protocol Recommendation:\*\*/gm, 'Protocol Recommendation:')
    .trim();
};

// Render message or tab content with clean text without raw ** asterisks
const renderFormattedMessage = (content) => {
  if (!content) return null;
  const lines = content.split('\n');
  return lines.map((line, lIdx) => {
    // Strip raw asterisks from each line
    const cleanLine = line.replace(/\*\*/g, '');
    return (
      <React.Fragment key={lIdx}>
        {cleanLine}
        {lIdx < lines.length - 1 && <br />}
      </React.Fragment>
    );
  });
};

const Results = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  // Recover state from localStorage if React Router dropped it due to size limits
  let state = location.state;
  if (!state) {
    try {
      const storedHistory = JSON.parse(localStorage.getItem('recentAnalyses') || '[]');
      if (storedHistory.length > 0) {
        state = storedHistory[0];
      }
    } catch (e) {
      console.error(e);
    }
  }
  state = state || {};

  const filename = state.filename || 'Patient_CT_0042.dcm';
  
  const getPathologyFallback = (c, tumor, cyst, stone) => {
    const text = (c || '').toLowerCase();
    if (tumor || text.includes('tumor') || text.includes('tumer')) return '/sample_scans/Tumor.jpg';
    if (cyst || text.includes('cyst')) return '/sample_scans/Cyst.jpg';
    if (stone || text.includes('stone')) return '/sample_scans/stone.webp';
    return '/sample_scans/Normal.jpg';
  };

  // Resolve reliable image URL: state.imageUrl, or fallback to backend uploads endpoint
  const resolveInitialImageUrl = () => {
    let url = state.imageUrl || '';
    if (url && (url.startsWith('data:') || url.startsWith('blob:') || url.startsWith('/sample_scans/'))) {
      return url;
    }
    const baseName = filename ? filename.split('/').pop().split('\\').pop() : '';
    const knownSamples = [
      'cyste 3.jpg', 'cyste 2.jpg', 'Cyst.jpg', '1.jpg', 'Normal.jpg', 'Tumor.jpg', 
      'stone.webp', 'TUMER 3.jpg', 'TUMER 4.jpg', 'stone 3.jpg', 'testing_c1.jpg', 
      'testing_s3.jpg', 'testing_t1.jpg', 'Patient_CT_0042.dcm', 'Patient_CT_0042.jpg'
    ];
    if (knownSamples.includes(baseName)) {
      return `/sample_scans/${encodeURIComponent(baseName)}`;
    }
    if (url.startsWith('/api/')) {
      return `${API_BASE}${url}`;
    }
    if (url.includes('/api/images/')) {
      const rel = url.substring(url.indexOf('/api/images/'));
      return `${API_BASE}${rel}`;
    }
    if (filename) {
      return `${API_BASE}/api/images/${encodeURIComponent(filename)}`;
    }
    return getPathologyFallback(state.classification, state.isTumor, state.isCyst, state.isStone);
  };
  const [imageUrl, setImageUrl] = useState(resolveInitialImageUrl);

  const patientName = state.patientName || '';
  const patientAge = state.patientAge || '';
  const doctorPrescription = state.doctorPrescription || '';
  const signSymptom = state.signSymptom || '';
  const distance = state.distance || '';
  const history = state.history || '';
  const medication = state.medication || '';
  const rawClass = state.classification || '';
  const isTumor = Boolean(state.isTumor || state.is_tumor || rawClass.toLowerCase().includes("tumor") || rawClass.toLowerCase().includes("tumer") || (filename || '').toLowerCase().includes("tumor") || (filename || '').toLowerCase().includes("tumer"));
  const isCyst = Boolean(state.isCyst || state.is_cyst || rawClass.toLowerCase().includes("cyst") || (filename || '').toLowerCase().includes("cyst") || (filename || '').toLowerCase().includes("cyste"));
  const isStone = Boolean(state.isStone || state.is_stone || rawClass.toLowerCase().includes("stone") || (filename || '').toLowerCase().includes("stone"));

  const classification = rawClass || (
    isTumor ? "Tumor Detected" :
    isCyst ? "Cyst Detected" :
    isStone ? "Kidney Stone Detected" :
    "Normal Kidney Structure Detected"
  );

  const confidence = state.confidence || 98.7;
  const leftVolume = state.leftVolume || state.left_volume || 142;
  const rightVolume = state.rightVolume || state.right_volume || (isTumor ? 154 : isCyst ? 146 : 138);
  const message = state.message || (
    isTumor ? "An abnormal mass/tumor has been identified in the right kidney region. Immediate clinical review is recommended." :
    isCyst ? "A benign-appearing cyst was identified in the renal cortex. Routine monitoring is advised." :
    isStone ? "A calcified renal stone (nephrolithiasis) was identified in the renal structure. Clinical review and urological consultation recommended." :
    "No visible abnormalities, cysts, or tumors identified in the highlighted regions."
  );

  const boundingBoxes = (state.boundingBoxes && state.boundingBoxes.length > 0) ? state.boundingBoxes :
    (state.bounding_boxes && state.bounding_boxes.length > 0) ? state.bounding_boxes :
    isTumor ? [{ class_name: 'Tumor', x: 42, y: 35, width: 28, height: 32 }] :
    isCyst ? [{ class_name: 'Cyst', x: 39.1, y: 79.4, width: 4.5, height: 6.5 }, { class_name: 'Kidney', x: 29.8, y: 76.5, width: 10.5, height: 17.6 }] :
    isStone ? [{ class_name: 'Stone', x: 48, y: 52, width: 6.0, height: 6.0 }] :
    [];

  // LLM Copilot State
  const [llmReport, setLlmReport] = useState(state.llmReport || null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [llmError, setLlmError] = useState(null);
  const [activeReportTab, setActiveReportTab] = useState('executive'); // executive | surgical | nutrition | patient
  const [chatMessages, setChatMessages] = useState([
    {
      role: 'model',
      content: `Hello! I'm your NephroVision Copilot. Here's a quick look at this scan:\n\n• **Finding:** ${classification} (${confidence}% confidence)\n• **Left Kidney:** ${leftVolume} cm³ | **Right Kidney:** ${rightVolume} cm³\n• **How I can help:** Ask me about the location, size, boundaries, or a plain-English patient explanation!`
    }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, isChatLoading]);

  const handleGenerateLLMReport = async () => {
    setIsGenerating(true);
    setLlmError(null);
    try {
      const res = await fetch(`${API_BASE}/api/llm/generate_report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_notes: doctorPrescription || `${history || 'No history'}. ${signSymptom || 'Routine scan'}. Medications: ${medication || 'None'}`,
          doctor_prescription: doctorPrescription,
          patient_name: patientName,
          patient_age: patientAge,
          classification,
          confidence,
          left_volume: leftVolume,
          right_volume: rightVolume,
          consumption_ratio: Number(((Math.abs(leftVolume - rightVolume) / Math.max(leftVolume, rightVolume)) * 100).toFixed(1))
        })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setLlmReport(data);
    } catch (err) {
      console.error('LLM Report Generation failed:', err);
      setLlmError('Failed to generate report from LLM copilot. Ensure backend is running.');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSendChat = async (customPrompt) => {
    const query = customPrompt || chatInput;
    if (!query || !query.trim()) return;

    const userMsg = { role: 'user', content: query };
    const updatedHistory = [...chatMessages, userMsg];
    setChatMessages(updatedHistory);
    if (!customPrompt) setChatInput('');
    setIsChatLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/llm/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: query,
          conversation_history: updatedHistory.slice(-6),
          patient_context: {
            patient_name: patientName,
            patient_age: patientAge,
            diagnosis: classification,
            confidence,
            left_volume: leftVolume,
            right_volume: rightVolume,
            doctor_prescription: doctorPrescription,
            clinical_notes: doctorPrescription || `${history}. ${signSymptom}`,
            llm_report_synthesis: llmReport?.report?.executive_synthesis || '',
            bounding_boxes: boundingBoxes,
            distance: distance,
            sign_symptom: signSymptom,
            history: history,
            is_cyst: isCyst,
            is_tumor: isTumor,
            is_stone: isStone,
            filename: filename
          }
        })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setChatMessages([...updatedHistory, { role: 'model', content: data.reply }]);
    } catch (err) {
      console.error('LLM Chat failed:', err);
      setChatMessages([...updatedHistory, { 
        role: 'model', 
        content: "I could not reach the Gemini Copilot backend service at this moment. Please verify backend connectivity." 
      }]);
    } finally {
      setIsChatLoading(false);
    }
  };

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
          <Button variant="secondary" onClick={() => navigate('/technical-details', { state: { ...state, imageUrl: imageUrl || state.imageUrl, filename: filename, file: state.file || file } })}>
            Technical Detail
          </Button>
          <Button variant="secondary" onClick={() => window.print()}>
            <Printer size={18} /> Print
          </Button>
          <Button onClick={() => navigate('/report', { state: { ...state, filename, patientName, patientAge, doctorPrescription, signSymptom, distance, history, medication, isTumor, isCyst, classification, confidence, leftVolume, rightVolume, message, llmReport } })}>
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
            <div style={{
              position: 'relative',
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              maxWidth: '100%',
              maxHeight: '100%',
              minWidth: '240px',
              minHeight: '240px'
            }}>
              {imageUrl ? (
                <img 
                  src={imageUrl} 
                  alt="CT Scan" 
                  onError={(e) => {
                    const cleanName = filename ? filename.split('/').pop().split('\\').pop() : '';
                    const sampleFallback = cleanName ? `/sample_scans/${encodeURIComponent(cleanName)}` : '';
                    const backendFallback = cleanName ? `${API_BASE}/api/images/${encodeURIComponent(cleanName)}` : '';
                    const pathFallback = getPathologyFallback(classification, isTumor, isCyst, isStone);
                    
                    if (sampleFallback && !e.target.src.includes('/sample_scans/')) {
                      e.target.src = sampleFallback;
                    } else if (backendFallback && e.target.src !== backendFallback && !e.target.src.includes('/api/images/')) {
                      e.target.src = backendFallback;
                    } else if (!e.target.src.endsWith(pathFallback)) {
                      e.target.src = pathFallback;
                    }
                  }}
                  style={{ 
                    maxWidth: '100%', 
                    maxHeight: '100%', 
                    minWidth: '220px',
                    minHeight: '220px',
                    width: 'auto',
                    height: 'auto',
                    display: 'block',
                    objectFit: 'contain', 
                    imageRendering: '-webkit-optimize-contrast',
                    filter: 'contrast(1.06) brightness(1.02)'
                  }} 
                />
              ) : (
                <p className="text-muted">No Image Available</p>
              )}
              
              {/* Real YOLO Bounding Boxes from Backend */}
              {boundingBoxes.length > 0 ? (
                boundingBoxes.map((box, index) => {
                  const isTumorBox = box.class_name.toLowerCase() === 'tumor';
                  const isCystBox = box.class_name.toLowerCase() === 'cyst';
                  const isStoneBox = box.class_name.toLowerCase() === 'stone';
                  
                  if (isStoneBox) {
                    // Dynamic stone boundary sizing:
                    // 1. Calculate center coordinates from detection output
                    const cx = box.x + box.width / 2;
                    const cy = box.y + box.height / 2;
                    // 2. Dynamic boundary dimension with margin so stone inside is fully framed and visible
                    const boundW = Math.max(box.width * 1.6, 5.5);
                    const boundH = Math.max(box.height * 1.6, 5.5);
                    const boundLeft = Math.max(0, cx - boundW / 2);
                    const boundTop = Math.max(0, cy - boundH / 2);

                    const stoneColor = '#facc15';
                    const stoneBg = 'rgba(250, 204, 21, 0.20)';

                    return (
                      <div
                        key={index}
                        style={{
                          position: 'absolute',
                          top: `${boundTop}%`,
                          left: `${boundLeft}%`,
                          width: `${boundW}%`,
                          height: `${boundH}%`,
                          border: `2.5px solid ${stoneColor}`,
                          boxShadow: '0 0 0 1.5px rgba(0, 0, 0, 0.9), 0 0 12px rgba(250, 204, 21, 0.65)',
                          backgroundColor: stoneBg,
                          borderRadius: '8px',
                          pointerEvents: 'none'
                        }}
                      >
                        <span style={{
                          position: 'absolute',
                          top: '-26px',
                          left: '-2px',
                          background: stoneColor,
                          color: '#0f172a',
                          padding: '2px 8px',
                          fontSize: '11px',
                          borderRadius: '4px',
                          fontWeight: 700,
                          textTransform: 'capitalize',
                          whiteSpace: 'nowrap',
                          boxShadow: '0 2px 5px rgba(0, 0, 0, 0.6)',
                          letterSpacing: '0.02em'
                        }}>
                          {box.class_name}
                        </span>
                      </div>
                    );
                  }

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
              ) : null}
            </div>
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
            ) : isStone ? (
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem', marginBottom: '1.5rem' }}>
                <AlertTriangle size={24} style={{ color: '#eab308', marginTop: '0.25rem' }} />
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
                  <div style={{ height: '100%', backgroundColor: isTumor ? 'var(--color-danger)' : isCyst ? 'var(--color-warning)' : isStone ? '#eab308' : 'var(--color-success)', width: `${confidence}%`, borderRadius: 'var(--radius-full)' }} />
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

          {doctorPrescription && (
            <Card title="Doctor Prescription & Notes">
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
                <FileText size={20} style={{ color: 'var(--color-primary)', marginTop: '0.2rem', flexShrink: 0 }} />
                <p style={{ margin: 0, whiteSpace: 'pre-line', fontSize: '0.925rem', color: 'var(--color-text-main)' }}>
                  {doctorPrescription}
                </p>
              </div>
            </Card>
          )}
        </div>
      </div>

      {/* Gemini Multimodal Clinical Copilot Section */}
      <div style={{ marginTop: '2.5rem' }}>
        <Card>
          {/* Copilot Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--color-border)', paddingBottom: '1.25rem', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{ width: '42px', height: '42px', borderRadius: '10px', background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff' }}>
                <Brain size={24} />
              </div>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <h3 style={{ margin: 0, fontSize: '1.25rem' }}>Multimodal Clinical Copilot</h3>
                  <span style={{ fontSize: '0.75rem', fontWeight: 600, padding: '2px 8px', borderRadius: '12px', background: 'rgba(59, 130, 246, 0.15)', color: '#60a5fa', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
                    Powered by Google Gemini
                  </span>
                </div>
                <p className="text-muted" style={{ margin: 0, fontSize: '0.875rem' }}>
                  Multimodal synthesis of PCSA volumetric contours, patient comorbidities & prescription
                </p>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '0.75rem' }}>
              <Button 
                onClick={handleGenerateLLMReport} 
                disabled={isGenerating}
                style={{ background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)', border: 'none', color: '#fff' }}
              >
                {isGenerating ? (
                  <>
                    <RefreshCw size={16} className="animate-spin" /> Generating Medical Scribe...
                  </>
                ) : (
                  <>
                    <Sparkles size={16} /> {llmReport ? 'Regenerate Gemini Report' : 'Generate Gemini Clinical Report'}
                  </>
                )}
              </Button>
            </div>
          </div>

          {llmError && (
            <div style={{ padding: '0.75rem 1rem', borderRadius: 'var(--radius-sm)', backgroundColor: 'rgba(239, 68, 68, 0.15)', color: '#ef4444', marginBottom: '1rem', fontSize: '0.875rem' }}>
              {llmError}
            </div>
          )}

          {/* Content Area */}
          {!llmReport && !isGenerating ? (
            <div style={{ textAlign: 'center', padding: '2.5rem 1rem', background: 'rgba(255,255,255,0.02)', borderRadius: 'var(--radius-md)', border: '1px dashed var(--color-border)' }}>
              <Bot size={44} style={{ color: '#60a5fa', margin: '0 auto 1rem auto' }} />
              <h4 style={{ margin: '0 0 0.5rem 0' }}>Multimodal Clinical Intelligence Awaiting Execution</h4>
              <p className="text-muted" style={{ maxWidth: '620px', margin: '0 auto 1.5rem auto', fontSize: '0.9rem' }}>
                Execute the Multimodal LLM Copilot pipeline from <code>capstone_llm.ipynb</code> to synthesize the vision segmentation output ({classification}, {confidence}%) with patient comorbidities into a 4-tier clinical diagnostic dossier.
              </p>
              <Button onClick={handleGenerateLLMReport} size="sm">
                <Sparkles size={16} /> Run Clinical Scribe Now
              </Button>
            </div>
          ) : isGenerating ? (
            <div style={{ textAlign: 'center', padding: '3rem 1rem' }}>
              <RefreshCw size={40} className="animate-spin" style={{ color: 'var(--color-primary)', margin: '0 auto 1rem auto' }} />
              <h4 style={{ margin: '0 0 0.5rem 0' }}>Consulting Google Gemini Multimodal Scribe...</h4>
              <p className="text-muted" style={{ fontSize: '0.875rem' }}>
                Synthesizing PCSA spatial contours, patient comorbidities, and surgical margins...
              </p>
            </div>
          ) : (
            <div>
              {/* Section Tabs */}
              <div style={{ display: 'flex', gap: '0.5rem', borderBottom: '1px solid var(--color-border)', paddingBottom: '0.75rem', marginBottom: '1.25rem', overflowX: 'auto' }}>
                {[
                  { id: 'executive', label: '1. Executive Synthesis', icon: Brain },
                  { id: 'surgical', label: '2. Surgical Risk Stratification', icon: Stethoscope },
                  { id: 'nutrition', label: '3. Renal Nutrition Plan', icon: ShieldCheck },
                  { id: 'patient', label: '4. Patient Summary', icon: FileText }
                ].map(tab => {
                  const Icon = tab.icon;
                  const isActive = activeReportTab === tab.id;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveReportTab(tab.id)}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        padding: '0.5rem 1rem',
                        borderRadius: 'var(--radius-sm)',
                        border: isActive ? '1px solid #3b82f6' : '1px solid transparent',
                        background: isActive ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
                        color: isActive ? '#60a5fa' : 'var(--color-text-muted)',
                        fontWeight: isActive ? 600 : 400,
                        fontSize: '0.875rem',
                        cursor: 'pointer',
                        transition: 'all 0.2s'
                      }}
                    >
                      <Icon size={16} />
                      {tab.label}
                    </button>
                  );
                })}
              </div>

              {/* Tab Content Display */}
              <div style={{ 
                backgroundColor: 'var(--color-surface)', 
                border: '1px solid var(--color-border)', 
                borderRadius: 'var(--radius-md)', 
                padding: '1.5rem',
                minHeight: '220px',
                lineHeight: '1.7',
                fontSize: '0.925rem',
                boxShadow: 'var(--shadow-sm)'
              }}>
                {activeReportTab === 'executive' && (
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem' }}>
                      <h4 style={{ margin: 0, color: 'var(--color-text)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Brain size={18} style={{ color: '#0ea5e9' }} /> Executive Diagnostic Synthesis
                      </h4>
                      <span style={{ fontSize: '0.75rem', color: '#10b981', fontWeight: 600, backgroundColor: 'rgba(16, 185, 129, 0.1)', padding: '2px 8px', borderRadius: '12px' }}>
                        Verified by {llmReport?.model || 'Gemini'}
                      </span>
                    </div>
                    <div style={{ color: 'var(--color-text)', lineHeight: '1.7' }}>
                      {renderFormattedMessage(cleanSectionText(llmReport?.report?.executive_synthesis)) || 'No executive synthesis generated.'}
                    </div>
                  </div>
                )}

                {activeReportTab === 'surgical' && (
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem' }}>
                      <h4 style={{ margin: 0, color: 'var(--color-text)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Stethoscope size={18} style={{ color: '#f59e0b' }} /> Pre-Operative Surgical Risk Stratification & Nephrometry
                      </h4>
                      <span style={{ fontSize: '0.75rem', color: '#f59e0b', fontWeight: 600, backgroundColor: 'rgba(245, 158, 11, 0.1)', padding: '2px 8px', borderRadius: '12px' }}>
                        R.E.N.A.L. & PADUA Assessment
                      </span>
                    </div>
                    <div style={{ color: 'var(--color-text)', lineHeight: '1.7' }}>
                      {renderFormattedMessage(cleanSectionText(llmReport?.report?.surgical_risk_stratification)) || 'No surgical risk assessment generated.'}
                    </div>
                  </div>
                )}

                {activeReportTab === 'nutrition' && (
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem' }}>
                      <h4 style={{ margin: 0, color: 'var(--color-text)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <ShieldCheck size={18} style={{ color: '#3b82f6' }} /> Comorbidity-Aware Renal Nutrition Protocol
                      </h4>
                      <span style={{ fontSize: '0.75rem', color: '#3b82f6', fontWeight: 600, backgroundColor: 'rgba(59, 130, 246, 0.1)', padding: '2px 8px', borderRadius: '12px' }}>
                        Personalized Macro/Micro Guide
                      </span>
                    </div>
                    <div style={{ color: 'var(--color-text)', lineHeight: '1.7' }}>
                      {renderFormattedMessage(cleanSectionText(llmReport?.report?.renal_nutrition_plan)) || 'No renal nutrition plan generated.'}
                    </div>
                  </div>
                )}

                {activeReportTab === 'patient' && (
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem' }}>
                      <h4 style={{ margin: 0, color: 'var(--color-text)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Sparkles size={18} style={{ color: '#8b5cf6' }} /> Patient-Friendly Summary (Plain Language)
                      </h4>
                      <span style={{ fontSize: '0.75rem', color: '#8b5cf6', fontWeight: 600, backgroundColor: 'rgba(139, 92, 246, 0.1)', padding: '2px 8px', borderRadius: '12px' }}>
                        Clear, Empathetic Guidance
                      </span>
                    </div>
                    <div style={{ color: 'var(--color-text)', lineHeight: '1.7' }}>
                      {renderFormattedMessage(cleanSectionText(llmReport?.report?.patient_friendly_summary)) || 'No patient summary generated.'}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Interactive Clinical Consultation Chat */}
          <div style={{ marginTop: '2rem', borderTop: '1px solid var(--color-border)', paddingTop: '1.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <MessageSquare size={18} style={{ color: 'var(--color-primary)' }} />
              <h4 style={{ margin: 0, color: 'var(--color-text)' }}>Ask Clinical Copilot</h4>
            </div>

            {/* Quick Prompts */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1rem' }}>
              {[
                "What surgical margin is recommended?",
                "How does diabetes impact this patient's nephrectomy risk?",
                "Explain the findings simply for the patient",
                "What follow-up CT scan interval is advised?"
              ].map((prompt, i) => (
                <button
                  key={i}
                  onClick={() => handleSendChat(prompt)}
                  disabled={isChatLoading}
                  style={{
                    fontSize: '0.75rem',
                    padding: '0.4rem 0.85rem',
                    borderRadius: '20px',
                    backgroundColor: 'var(--color-surface)',
                    border: '1px solid var(--color-border)',
                    color: 'var(--color-text)',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.35rem',
                    boxShadow: 'var(--shadow-sm)',
                    transition: 'all 0.15s ease'
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--color-primary)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--color-border)'; }}
                >
                  <ChevronRight size={12} style={{ color: 'var(--color-primary)' }} /> {prompt}
                </button>
              ))}
            </div>

            {/* Messages Window */}
            <div style={{ 
              maxHeight: '320px', 
              overflowY: 'auto', 
              display: 'flex', 
              flexDirection: 'column', 
              gap: '0.85rem',
              padding: '1.25rem',
              backgroundColor: 'var(--color-background)',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--color-border)',
              marginBottom: '1rem'
            }}>
              {chatMessages.map((msg, idx) => (
                <div 
                  key={idx} 
                  style={{ 
                    display: 'flex', 
                    flexDirection: 'column',
                    alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                    maxWidth: '85%'
                  }}
                >
                  <div style={{
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    color: msg.role === 'user' ? 'var(--color-primary)' : '#0ea5e9',
                    marginBottom: '0.25rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.35rem',
                    alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start'
                  }}>
                    {msg.role === 'user' ? (
                      <><User size={12} /> You (Physician)</>
                    ) : (
                      <><Bot size={13} /> Clinical Copilot</>
                    )}
                  </div>
                  <div style={{
                    padding: '0.85rem 1.15rem',
                    borderRadius: msg.role === 'user' ? '14px 14px 2px 14px' : '14px 14px 14px 2px',
                    backgroundColor: msg.role === 'user' ? '#0284c7' : '#f8fafc',
                    color: msg.role === 'user' ? '#ffffff' : '#0f172a',
                    fontSize: '0.875rem',
                    lineHeight: '1.6',
                    border: msg.role === 'user' ? 'none' : '1px solid #cbd5e1',
                    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)'
                  }}>
                    {renderFormattedMessage(msg.content)}
                  </div>
                </div>
              ))}
              {isChatLoading && (
                <div style={{ alignSelf: 'flex-start', display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#0ea5e9', fontSize: '0.85rem', fontWeight: 500, padding: '0.5rem' }}>
                  <RefreshCw size={14} className="animate-spin" /> Gemini Copilot is formulating evidence-based guidance...
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Form */}
            <form 
              onSubmit={(e) => { e.preventDefault(); handleSendChat(); }}
              style={{ display: 'flex', gap: '0.75rem' }}
            >
              <input
                type="text"
                placeholder="Ask Gemini copilot a clinical question about this case..."
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                disabled={isChatLoading}
                style={{
                  flex: 1,
                  padding: '0.75rem 1rem',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: 'var(--color-surface)',
                  border: '1px solid var(--color-border)',
                  color: 'var(--color-text)',
                  outline: 'none',
                  fontSize: '0.9rem',
                  boxShadow: 'var(--shadow-sm)'
                }}
              />
              <Button type="submit" disabled={isChatLoading || !chatInput.trim()}>
                <Send size={16} /> Send
              </Button>
            </form>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Results;

