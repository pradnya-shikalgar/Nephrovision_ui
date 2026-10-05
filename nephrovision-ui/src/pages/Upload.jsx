import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, ArrowLeft, User } from 'lucide-react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import FileUpload from '../components/ui/FileUpload';

const Upload = () => {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [patientName, setPatientName] = useState('');
  const [patientAge, setPatientAge] = useState('');

  const handleFileSelect = (file) => {
    setSelectedFile(file);
  };

  const handleAnalyze = () => {
    if (selectedFile) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const imageUrl = reader.result;
        navigate('/analyze', { state: { file: selectedFile, filename: selectedFile.name, imageUrl, patientName, patientAge } });
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
