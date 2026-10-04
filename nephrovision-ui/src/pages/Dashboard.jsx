import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, Clock, ShieldCheck, ArrowRight } from 'lucide-react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import './Dashboard.css';

const Dashboard = () => {
  const navigate = useNavigate();
  const [recentAnalyses, setRecentAnalyses] = useState(() => {
    return JSON.parse(localStorage.getItem('recentAnalyses') || '[]');
  });

  const updateStatus = (index, newStatus) => {
    const updated = [...recentAnalyses];
    updated[index].statusOverride = newStatus;
    setRecentAnalyses(updated);
    localStorage.setItem('recentAnalyses', JSON.stringify(updated));
  };

  const deleteAnalysis = async (index, filename) => {
    try {
      // Call backend to delete (mock endpoint for now)
      await fetch(`http://127.0.0.1:8000/api/analyze/${filename}`, {
        method: 'DELETE',
      });
    } catch (e) {
      console.error("Failed to delete from backend", e);
    }
    // Delete from frontend state and local storage
    const updated = [...recentAnalyses];
    updated.splice(index, 1);
    setRecentAnalyses(updated);
    localStorage.setItem('recentAnalyses', JSON.stringify(updated));
  };

  return (
    <div className="page-container animate-fade-in">
      <div className="dashboard-header mb-8">
        <div>
          <h1 className="mb-4">NephroVision Analysis Platform</h1>
          <p className="text-muted">Advanced Kidney Segmentation & Anomaly Detection using Deep Learning.</p>
        </div>
        <Button size="lg" onClick={() => navigate('/upload')}>
          New Analysis <ArrowRight size={18} />
        </Button>
      </div>

      <div className="stats-grid mb-8">
        <Card className="stat-card">
          <div className="stat-icon-wrapper blue">
            <Activity size={24} />
          </div>
          <div className="stat-content">
            <p className="text-muted">Total Scans</p>
            <h3>1,248</h3>
          </div>
        </Card>
        <Card className="stat-card">
          <div className="stat-icon-wrapper green">
            <ShieldCheck size={24} />
          </div>
          <div className="stat-content">
            <p className="text-muted">Accuracy</p>
            <h3>98.2%</h3>
          </div>
        </Card>
        <Card className="stat-card">
          <div className="stat-icon-wrapper orange">
            <Clock size={24} />
          </div>
          <div className="stat-content">
            <p className="text-muted">Avg. Processing Time</p>
            <h3>2.4s</h3>
          </div>
        </Card>
      </div>

      <Card title="Recent Analyses">
        <div className="recent-list">
          {recentAnalyses.length > 0 ? (
            recentAnalyses.map((analysis, index) => (
              <div 
                key={index} 
                className="recent-item" 
                style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                onClick={() => navigate('/results', { state: analysis })}
              >
                <div className="recent-info">
                  <h4>{analysis.patientName ? `${analysis.patientName} (${analysis.filename})` : analysis.filename}</h4>
                  <p className="text-muted">{analysis.date} - {analysis.classification}</p>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <div style={{ display: 'flex', gap: '0.5rem' }} onClick={(e) => e.stopPropagation()}>
                    <button 
                      onClick={() => {
                        updateStatus(index, 'Review');
                        navigate('/results', { state: analysis });
                      }}
                      style={{ padding: '4px 8px', fontSize: '0.75rem', borderRadius: '4px', border: '1px solid var(--color-warning)', background: 'transparent', color: 'var(--color-warning)', cursor: 'pointer' }}
                    >
                      Review
                    </button>
                    <button 
                      onClick={() => {
                        updateStatus(index, 'Completed');
                        navigate('/report', { state: analysis });
                      }}
                      style={{ padding: '4px 8px', fontSize: '0.75rem', borderRadius: '4px', border: '1px solid var(--color-success)', background: 'transparent', color: 'var(--color-success)', cursor: 'pointer' }}
                    >
                      Complete
                    </button>
                    <button 
                      onClick={() => deleteAnalysis(index, analysis.filename)}
                      style={{ padding: '4px 8px', fontSize: '0.75rem', borderRadius: '4px', border: '1px solid var(--color-danger)', background: 'transparent', color: 'var(--color-danger)', cursor: 'pointer' }}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--color-text-muted)' }}>
              <p>No recent analyses found.</p>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export default Dashboard;
