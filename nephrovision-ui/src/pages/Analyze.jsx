import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Loader2 } from 'lucide-react';
import Card from '../components/ui/Card';

const Analyze = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const file = location.state?.file;
  const imageUrl = location.state?.imageUrl;
  const patientName = location.state?.patientName || '';
  const patientAge = location.state?.patientAge || '';
  const filename = location.state?.filename || 'Patient_CT_0042.dcm';
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!file) {
      navigate('/upload');
      return;
    }

    // Fake progress bar while uploading/analyzing
    const interval = setInterval(() => {
      setProgress(p => (p >= 90 ? 90 : p + Math.random() * 15));
    }, 500);

    const performAnalysis = async () => {
      try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('http://localhost:8000/api/analyze', {
          method: 'POST',
          body: formData
        });
        
        if (!response.ok) throw new Error("API Request Failed");
        const data = await response.json();
        
        clearInterval(interval);
        setProgress(100);
        
        setTimeout(() => {
          const analysisResult = { 
            filename, 
            imageUrl,
            patientName,
            patientAge,
            signSymptom: data.sign_symptom,
            distance: data.distance,
            history: data.history,
            medication: data.medication,
            isTumor: data.is_tumor,
            isCyst: data.is_cyst,
            classification: data.classification,
            confidence: data.confidence,
            leftVolume: data.left_volume,
            rightVolume: data.right_volume,
            message: data.message,
            boundingBoxes: data.bounding_boxes || [],
            date: new Date().toLocaleString()
          };

          const existingHistory = JSON.parse(localStorage.getItem('recentAnalyses') || '[]');
          localStorage.setItem('recentAnalyses', JSON.stringify([analysisResult, ...existingHistory].slice(0, 10)));

          navigate('/results', { state: { ...analysisResult, file } });
        }, 500);
      } catch (err) {
        console.error(err);
        clearInterval(interval);
        setError("Analysis failed to connect to backend server. Make sure FastAPI is running on port 8000.");
      }
    };

    performAnalysis();

    return () => clearInterval(interval);
  }, [navigate, file, filename, imageUrl, patientName, patientAge]);

  return (
    <div className="page-container animate-fade-in" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '80vh' }}>
      <Card className="text-center" style={{ maxWidth: '500px', width: '100%', padding: '2rem' }}>
        {error ? (
          <>
             <h2 className="mb-4" style={{ color: 'var(--color-danger)' }}>Error</h2>
             <p className="text-muted">{error}</p>
             <button onClick={() => navigate('/upload')} style={{ marginTop: '1rem', padding: '0.5rem 1rem' }}>Go Back</button>
          </>
        ) : (
          <>
            <Loader2 className="animate-spin" size={64} style={{ color: 'var(--color-primary)', margin: '0 auto 1.5rem' }} />
            <h2 className="mb-4">Analyzing CT Scan...</h2>
            <p className="text-muted mb-8">Running NephroVision Deep Learning Model.</p>
        
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
          </>
        )}
      </Card>
    </div>
  );
};

export default Analyze;
