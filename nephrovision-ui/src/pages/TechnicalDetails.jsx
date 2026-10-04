import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, Loader2 } from 'lucide-react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';

const TechnicalDetails = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const file = location.state?.file;
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [images, setImages] = useState(null);

  useEffect(() => {
    if (!file) {
      setError("This analysis was loaded from the dashboard history. To view dynamic Technical Details (which requires running the live AI architecture on the image), please upload the CT scan again through the 'New Analysis' flow.");
      setLoading(false);
      return;
    }

    const fetchTechnicalDetails = async () => {
      const formData = new FormData();
      formData.append('file', file);
      
      try {
        const response = await fetch('http://127.0.0.1:8000/api/technical_details', {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) throw new Error("API Request Failed");
        const data = await response.json();
        setImages(data);
        setLoading(false);
      } catch (err) {
        console.error(err);
        setError("Failed to fetch technical details. Ensure backend is running.");
        setLoading(false);
      }
    };

    fetchTechnicalDetails();
  }, [file]);

  return (
    <div className="page-container animate-fade-in">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '2rem' }}>
        <Button variant="secondary" size="sm" onClick={() => navigate(-1)}>
          <ArrowLeft size={16} /> Back to Results
        </Button>
        <h2 style={{ margin: 0 }}>Advanced Technical Details</h2>
        <div style={{ width: '120px' }}></div> {/* Spacer for center alignment */}
      </div>

      {loading && (
        <Card style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '50vh' }}>
          <Loader2 size={48} className="animate-spin text-primary mb-4" />
          <h3 className="text-muted">Generating AI Diagnostics...</h3>
          <p className="text-muted text-sm mt-2">Running Grad-CAM and Surgical Boundary Algorithms. This may take a few seconds.</p>
        </Card>
      )}

      {error && (
        <Card style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '50vh', color: 'var(--color-danger)' }}>
          <h3>Error</h3>
          <p>{error}</p>
        </Card>
      )}

      {images && !loading && !error && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
          <Card title="1. Original CT Scan" style={{ textAlign: 'center' }}>
            <img src={`data:image/jpeg;base64,${images.original_image}`} alt="Original" style={{ maxWidth: '100%', borderRadius: 'var(--radius-md)' }} />
          </Card>
          <Card title="2. PCSA Attention Map" style={{ textAlign: 'center' }}>
            <img src={`data:image/jpeg;base64,${images.gradcam_image}`} alt="Grad-CAM" style={{ maxWidth: '100%', borderRadius: 'var(--radius-md)' }} />
          </Card>
          <Card title="3. Surgical Boundaries" style={{ textAlign: 'center' }}>
            <img src={`data:image/jpeg;base64,${images.surgical_boundaries}`} alt="Surgical" style={{ maxWidth: '100%', borderRadius: 'var(--radius-md)' }} />
          </Card>
          <Card title="4. Surgical Impact Ratio" style={{ textAlign: 'center' }}>
            <img src={`data:image/jpeg;base64,${images.impact_ratio}`} alt="Impact Ratio" style={{ maxWidth: '100%', borderRadius: 'var(--radius-md)' }} />
          </Card>
        </div>
      )}
    </div>
  );
};

export default TechnicalDetails;
