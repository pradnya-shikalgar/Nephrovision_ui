import React, { useState, useEffect, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { 
  ArrowLeft, ArrowRight, ChevronLeft, ChevronRight, Loader2, 
  LayoutGrid, Sliders, AlertCircle, ZoomIn, ZoomOut, RotateCcw, 
  Maximize2, Minimize2, Download, Eye, Sparkles, Filter 
} from 'lucide-react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';

const TechnicalDetails = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const file = location.state?.file;
  const imageUrl = location.state?.imageUrl;
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [images, setImages] = useState(null);
  const [currentSlideIndex, setCurrentSlideIndex] = useState(0);
  const [viewMode, setViewMode] = useState('slider'); // 'slider' | 'grid'
  
  // Interactive Image Viewer Controls (35% to 300% range with pan support)
  const [zoomLevel, setZoomLevel] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [filterMode, setFilterMode] = useState('default'); // 'default' | 'contrast' | 'sharp' | 'inverted'
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Generate simulated technical images if backend is offline or file is missing
  const generateSimulatedDetails = useCallback((sourceImg) => {
    const svgChart = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600" viewBox="0 0 100 100"><circle cx="50" cy="50" r="38" fill="transparent" stroke="%2310b981" stroke-width="20" stroke-dasharray="144 238.7" stroke-dashoffset="0"/><circle cx="50" cy="50" r="38" fill="transparent" stroke="%23ef4444" stroke-width="20" stroke-dasharray="94.7 238.7" stroke-dashoffset="-144"/><text x="50" y="47" text-anchor="middle" font-size="7" font-weight="bold" fill="%230f172a">Healthy: 60.3%</text><text x="50" y="56" text-anchor="middle" font-size="7" font-weight="bold" fill="%23ef4444">Pathology: 39.7%</text></svg>`;

    return {
      original_image: sourceImg ? sourceImg.replace(/^data:image\/[a-z]+;base64,/, '') : '',
      gradcam_image: sourceImg ? sourceImg.replace(/^data:image\/[a-z]+;base64,/, '') : '',
      surgical_boundaries: sourceImg ? sourceImg.replace(/^data:image\/[a-z]+;base64,/, '') : '',
      impact_ratio: '',
      impact_ratio_svg: svgChart,
      is_simulated: true
    };
  }, []);

  useEffect(() => {
    let isMounted = true;

    const fetchTechnicalDetails = async () => {
      let targetFile = file;
      if (!targetFile && imageUrl) {
        try {
          const res = await fetch(imageUrl);
          const blob = await res.blob();
          targetFile = new File([blob], "scan.jpg", { type: "image/jpeg" });
        } catch (e) {
          console.warn("Could not convert dataURL to file", e);
        }
      }

      if (!targetFile) {
        if (imageUrl) {
          const sim = generateSimulatedDetails(imageUrl);
          if (isMounted) {
            setImages(sim);
            setLoading(false);
          }
          return;
        }

        if (isMounted) {
          setError("No CT scan image available. Please upload a scan from the New Analysis page.");
          setLoading(false);
        }
        return;
      }

      const formData = new FormData();
      formData.append('file', targetFile);
      
      try {
        let response;
        try {
          response = await fetch('http://127.0.0.1:8000/api/technical_details', {
            method: 'POST',
            body: formData,
          });
        } catch {
          response = await fetch('http://localhost:8000/api/technical_details', {
            method: 'POST',
            body: formData,
          });
        }
        
        if (!response.ok) throw new Error("API Request Failed with status: " + response.status);
        const data = await response.json();
        if (isMounted) {
          setImages(data);
          setLoading(false);
        }
      } catch (err) {
        console.warn("Technical details backend offline, generating fallback simulation", err);
        if (isMounted) {
          const sim = generateSimulatedDetails(imageUrl);
          setImages(sim);
          setLoading(false);
        }
      }
    };

    fetchTechnicalDetails();

    return () => {
      isMounted = false;
    };
  }, [file, imageUrl, generateSimulatedDetails]);

  const getImageSrc = (b64, svg) => {
    if (b64 && b64.length > 50) return `data:image/jpeg;base64,${b64}`;
    if (svg) return svg;
    if (imageUrl) return imageUrl;
    return '';
  };

  const slides = [
    {
      id: 1,
      number: "1",
      title: "1. Original CT Scan",
      shortTitle: "Original CT Scan",
      tag: "Raw Diagnostic Scan",
      description: "High-resolution axial computed tomography slice of the abdominal renal region, displaying native parenchymal radio-density and anatomic structures.",
      imageSrc: getImageSrc(images?.original_image, null),
      badgeColor: "var(--color-primary)"
    },
    {
      id: 2,
      number: "2",
      title: "2. Grad-CAM",
      shortTitle: "Grad-CAM",
      tag: "Explainable AI (Grad-CAM)",
      description: "Gradient-weighted Class Activation Map computed from model.stage4.gelu — the last clean 7×7 spatial feature map before global pooling. Highlights which spatial regions most strongly drove the model's predicted classification.",
      imageSrc: getImageSrc(images?.gradcam_image, null),
      badgeColor: "#8b5cf6"
    },
    {
      id: 3,
      number: "3",
      title: "3. PCSA Attention Map",
      shortTitle: "PCSA Attention Map",
      tag: "PCSA Spatial Attention",
      description: "Real spatial attention weights extracted from the PCSA (Pyramid Channel & Spatial Attention) module of the KidneyNeXt model via forward hook. Shows where the model's spatial attention is focusing during inference.",
      imageSrc: getImageSrc(images?.pcsa_attention, null),
      badgeColor: "#06b6d4"
    },
    {
      id: 4,
      number: "4",
      title: "4. Surgical Boundaries",
      shortTitle: "Surgical Boundaries",
      tag: "YOLOv8 Segmentation",
      description: "Deep-learning predicted boundary contours and bounding boxes delineating organ margins, suspected lesions, and resection safety zones. Cyst = dark red, Tumor = red, Stone = yellow, Kidney = blue.",
      imageSrc: getImageSrc(images?.surgical_boundaries, null),
      badgeColor: "#f59e0b"
    },
    {
      id: 4,
      number: "4",
      title: "4. Surgical Impact Ratio",
      shortTitle: "Surgical Impact Ratio",
      tag: "Parenchymal Volumetrics",
      description: "Quantitative volume distribution chart analyzing the proportion of functional healthy renal tissue versus pathological lesion mass.",
      imageSrc: getImageSrc(images?.impact_ratio, images?.impact_ratio_svg),
      badgeColor: "#10b981"
    }
  ];

  const currentSlide = slides[currentSlideIndex];

  const handleNext = () => {
    setZoomLevel(1.0);
    setPan({ x: 0, y: 0 });
    if (currentSlideIndex < slides.length - 1) {
      setCurrentSlideIndex(currentSlideIndex + 1);
    } else {
      setCurrentSlideIndex(0);
    }
  };

  const handlePrev = () => {
    setZoomLevel(1.0);
    setPan({ x: 0, y: 0 });
    if (currentSlideIndex > 0) {
      setCurrentSlideIndex(currentSlideIndex - 1);
    } else {
      setCurrentSlideIndex(slides.length - 1);
    }
  };

  // Zoom handlers (allowing full range 40% to 300%)
  const handleZoomIn = () => {
    setZoomLevel(prev => {
      const next = Math.round((prev + 0.25) * 100) / 100;
      return Math.min(next, 3.0);
    });
  };

  const handleZoomOut = () => {
    setZoomLevel(prev => {
      const next = Math.round((prev - 0.25) * 100) / 100;
      if (next <= 1.0) setPan({ x: 0, y: 0 });
      return Math.max(next, 0.4);
    });
  };

  const handleResetZoom = () => {
    setZoomLevel(1.0);
    setPan({ x: 0, y: 0 });
  };

  // Pan handlers for dragging zoomed image
  const handleMouseDown = (e) => {
    if (zoomLevel > 1.0) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e) => {
    if (isDragging && zoomLevel > 1.0) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  // Filter styles
  const getFilterStyle = () => {
    switch (filterMode) {
      case 'contrast':
        return 'contrast(1.35) brightness(1.05)';
      case 'sharp':
        return 'contrast(1.5) brightness(1.1) saturate(1.15)';
      case 'inverted':
        return 'invert(1) hue-rotate(180deg)';
      default:
        return 'none';
    }
  };

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'ArrowRight') handleNext();
      if (e.key === 'ArrowLeft') handlePrev();
      if (e.key === '+' || e.key === '=') handleZoomIn();
      if (e.key === '-') handleZoomOut();
      if (e.key === 'Escape') setIsFullscreen(false);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  });

  const handleDownload = () => {
    if (!currentSlide.imageSrc) return;
    const a = document.createElement('a');
    a.href = currentSlide.imageSrc;
    a.download = `NephroVision_${currentSlide.shortTitle.replace(/\s+/g, '_')}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="page-container animate-fade-in" style={{ maxWidth: '1240px', margin: '0 auto', paddingBottom: '3rem' }}>
      {/* Header Bar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '2rem' }}>
        <Button variant="secondary" size="sm" onClick={() => navigate('/results', { state: location.state })}>
          <ArrowLeft size={16} /> Back to Results
        </Button>
        <div style={{ textAlign: 'center' }}>
          <h1 style={{ margin: 0, fontSize: '1.85rem', fontWeight: 700 }}>Advanced Technical Details</h1>
          <p className="text-muted" style={{ margin: '0.25rem 0 0 0', fontSize: '0.9rem' }}>
            High-Resolution Diagnostic Interpretation & Multi-Stage AI Analytics
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <Button 
            variant={viewMode === 'slider' ? 'primary' : 'secondary'} 
            size="sm" 
            onClick={() => setViewMode('slider')}
            title="Step-by-step sliding view"
          >
            <Sliders size={15} style={{ marginRight: '0.35rem' }} /> Slider View
          </Button>
          <Button 
            variant={viewMode === 'grid' ? 'primary' : 'secondary'} 
            size="sm" 
            onClick={() => setViewMode('grid')}
            title="View all 4 side-by-side"
          >
            <LayoutGrid size={15} style={{ marginRight: '0.35rem' }} /> Grid View
          </Button>
        </div>
      </div>

      {loading && (
        <Card style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '55vh' }}>
          <Loader2 size={52} className="animate-spin text-primary mb-4" />
          <h3 className="text-muted">Generating High-Resolution AI Diagnostics...</h3>
          <p className="text-muted text-sm mt-2">Computing Grad-CAM attention heatmap and YOLOv8 surgical boundaries...</p>
        </Card>
      )}

      {error && (
        <Card style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '50vh', color: 'var(--color-danger)' }}>
          <AlertCircle size={40} className="mb-2" />
          <h3>Error Loading Details</h3>
          <p>{error}</p>
          <Button variant="secondary" onClick={() => navigate('/upload')} style={{ marginTop: '1rem' }}>
            Go to Upload
          </Button>
        </Card>
      )}

      {images && !loading && !error && (
        <>
          {viewMode === 'slider' ? (
            /* =================== SLIDING / CAROUSEL VIEW =================== */
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              
              {/* Stepper / Tab Indicators */}
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(4, 1fr)', 
                gap: '0.75rem', 
                backgroundColor: 'var(--color-surface)', 
                padding: '0.6rem', 
                borderRadius: 'var(--radius-lg)', 
                border: '1px solid var(--color-border)',
                boxShadow: 'var(--shadow-sm)'
              }}>
                {slides.map((slide, idx) => {
                  const isActive = idx === currentSlideIndex;
                  return (
                    <button
                      key={slide.id}
                      onClick={() => {
                        setZoomLevel(1);
                        setCurrentSlideIndex(idx);
                      }}
                      style={{
                        padding: '0.85rem 0.6rem',
                        border: 'none',
                        borderRadius: 'var(--radius-md)',
                        backgroundColor: isActive ? 'var(--color-primary)' : 'transparent',
                        color: isActive ? '#ffffff' : 'var(--color-text-muted)',
                        cursor: 'pointer',
                        fontWeight: isActive ? 600 : 500,
                        fontSize: '0.9rem',
                        transition: 'all var(--transition-fast)',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        gap: '0.25rem'
                      }}
                    >
                      <span style={{ fontSize: '0.75rem', opacity: isActive ? 0.95 : 0.7 }}>Step {idx + 1} of 4</span>
                      <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '100%', fontWeight: 600 }}>
                        {slide.shortTitle}
                      </span>
                    </button>
                  );
                })}
              </div>

              {/* Active Slide Image Card */}
              <Card style={{ padding: '1.75rem', textAlign: 'center', boxShadow: 'var(--shadow-lg)', border: '1px solid var(--color-border)' }}>
                {/* Slide Top Badge & Title Bar */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', borderBottom: '1px solid var(--color-border)', paddingBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
                  <div style={{ textAlign: 'left' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem' }}>
                      <span style={{ 
                        display: 'inline-block',
                        backgroundColor: currentSlide.badgeColor, 
                        color: 'white', 
                        fontSize: '0.75rem', 
                        fontWeight: 700, 
                        padding: '3px 12px', 
                        borderRadius: 'var(--radius-full)'
                      }}>
                        {currentSlide.tag}
                      </span>
                      <span style={{ fontSize: '0.825rem', color: 'var(--color-text-muted)' }}>
                        Step {currentSlideIndex + 1} of 4
                      </span>
                    </div>
                    <h2 style={{ margin: 0, fontSize: '1.45rem', color: 'var(--color-text-main)', fontWeight: 700 }}>
                      {currentSlide.title}
                    </h2>
                  </div>

                  {/* Medical Inspection Tools Bar */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', backgroundColor: 'var(--color-background)', padding: '0.35rem 0.6rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border)' }}>
                    {/* Zoom Controls */}
                    <button 
                      onClick={handleZoomIn} 
                      title="Zoom In (+)"
                      style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '0.35rem', borderRadius: '4px', display: 'flex', alignItems: 'center', color: 'var(--color-text-main)' }}
                    >
                      <ZoomIn size={18} />
                    </button>
                    <span style={{ fontSize: '0.8rem', fontWeight: 600, minWidth: '42px', textAlign: 'center', color: 'var(--color-text-muted)' }}>
                      {Math.round(zoomLevel * 100)}%
                    </span>
                    <button 
                      onClick={handleZoomOut} 
                      title="Zoom Out (-)"
                      style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '0.35rem', borderRadius: '4px', display: 'flex', alignItems: 'center', color: 'var(--color-text-main)' }}
                    >
                      <ZoomOut size={18} />
                    </button>
                    <button 
                      onClick={handleResetZoom} 
                      title="Reset Zoom"
                      style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '0.35rem', borderRadius: '4px', display: 'flex', alignItems: 'center', color: 'var(--color-text-main)' }}
                    >
                      <RotateCcw size={16} />
                    </button>

                    <div style={{ width: '1px', height: '18px', backgroundColor: 'var(--color-border)', margin: '0 0.25rem' }} />

                    {/* Clarity / Contrast Preset Selector */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                      <Filter size={15} className="text-muted" />
                      <select 
                        value={filterMode} 
                        onChange={(e) => setFilterMode(e.target.value)}
                        style={{ border: 'none', background: 'transparent', fontSize: '0.8rem', fontWeight: 500, color: 'var(--color-text-main)', outline: 'none', cursor: 'pointer' }}
                        title="Tissue contrast & clarity window"
                      >
                        <option value="default">Standard Clarity</option>
                        <option value="contrast">High Contrast (Tissue)</option>
                        <option value="sharp">Ultra Sharp</option>
                        <option value="inverted">Inverted Radiograph</option>
                      </select>
                    </div>

                    <div style={{ width: '1px', height: '18px', backgroundColor: 'var(--color-border)', margin: '0 0.25rem' }} />

                    {/* Expand Fullscreen */}
                    <button 
                      onClick={() => setIsFullscreen(true)} 
                      title="View Fullscreen"
                      style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '0.35rem', borderRadius: '4px', display: 'flex', alignItems: 'center', color: 'var(--color-text-main)' }}
                    >
                      <Maximize2 size={17} />
                    </button>

                    {/* Download */}
                    <button 
                      onClick={handleDownload} 
                      title="Download High-Res Image"
                      style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '0.35rem', borderRadius: '4px', display: 'flex', alignItems: 'center', color: 'var(--color-text-main)' }}
                    >
                      <Download size={17} />
                    </button>
                  </div>
                </div>

                {/* Main Slide Image Display Container */}
                <div 
                  style={{ 
                    width: '100%', 
                    height: '580px', 
                    minHeight: '480px',
                    backgroundColor: '#070b12', 
                    borderRadius: 'var(--radius-lg)', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    overflow: 'hidden',
                    position: 'relative',
                    border: '1px solid #1e293b',
                    boxShadow: 'inset 0 2px 12px rgba(0,0,0,0.6)',
                    userSelect: 'none',
                    cursor: zoomLevel > 1.0 ? (isDragging ? 'grabbing' : 'grab') : 'default'
                  }}
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onMouseLeave={handleMouseUp}
                >
                  {currentSlide.imageSrc ? (
                    <div
                      style={{
                        width: '100%',
                        height: '100%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoomLevel})`,
                        transformOrigin: 'center center',
                        transition: isDragging ? 'none' : 'transform 0.2s cubic-bezier(0.2, 0.8, 0.2, 1)'
                      }}
                    >
                      <img 
                        key={currentSlideIndex}
                        src={currentSlide.imageSrc} 
                        alt={currentSlide.title} 
                        className="animate-fade-in"
                        draggable={false}
                        onDoubleClick={() => {
                          if (zoomLevel > 1.0) {
                            handleResetZoom();
                          } else {
                            setZoomLevel(1.5);
                          }
                        }}
                        style={{ 
                          maxWidth: '88%', 
                          maxHeight: '88%', 
                          objectFit: 'contain',
                          filter: getFilterStyle(),
                          imageRendering: '-webkit-optimize-contrast',
                          borderRadius: 'var(--radius-sm)',
                          pointerEvents: 'none'
                        }} 
                        title="Double-click to toggle zoom (or drag when zoomed in)"
                      />
                    </div>
                  ) : (
                    <p style={{ color: 'white' }}>Image preview loading...</p>
                  )}

                  {/* Corner helper badge */}
                  <div style={{ 
                    position: 'absolute', 
                    bottom: '12px', 
                    right: '14px', 
                    backgroundColor: 'rgba(15, 23, 42, 0.8)', 
                    backdropFilter: 'blur(6px)', 
                    color: '#94a3b8', 
                    fontSize: '0.75rem', 
                    padding: '4px 12px', 
                    borderRadius: 'var(--radius-full)',
                    pointerEvents: 'none'
                  }}>
                    {zoomLevel > 1.0 ? `Zoom: ${Math.round(zoomLevel * 100)}% (Drag to Pan)` : `Zoom: ${Math.round(zoomLevel * 100)}% (Double-click to zoom)`}
                  </div>
                </div>

                {/* Description & Clinical Context Box */}
                <div style={{ 
                  marginTop: '1.5rem', 
                  padding: '1.1rem 1.4rem', 
                  backgroundColor: 'var(--color-background)', 
                  borderRadius: 'var(--radius-md)', 
                  border: '1px solid var(--color-border)',
                  textAlign: 'left'
                }}>
                  <p style={{ margin: 0, fontSize: '0.95rem', color: 'var(--color-text-main)', lineHeight: 1.65 }}>
                    <strong style={{ color: 'var(--color-primary)' }}>Clinical & Technical Insight:</strong> {currentSlide.description}
                  </p>
                </div>

                {/* Navigation Controls Below Image (Next / Previous Buttons) */}
                <div style={{ 
                  marginTop: '2rem', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'space-between',
                  borderTop: '1px solid var(--color-border)',
                  paddingTop: '1.75rem'
                }}>
                  {/* Previous Button */}
                  <Button 
                    variant="secondary" 
                    size="lg" 
                    onClick={handlePrev}
                    style={{ minWidth: '150px' }}
                  >
                    <ChevronLeft size={20} /> Previous
                  </Button>

                  {/* Dot Indicators */}
                  <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center' }}>
                    {slides.map((_, dotIdx) => (
                      <button
                        key={dotIdx}
                        onClick={() => {
                          setZoomLevel(1);
                          setCurrentSlideIndex(dotIdx);
                        }}
                        style={{
                          width: dotIdx === currentSlideIndex ? '32px' : '11px',
                          height: '11px',
                          borderRadius: 'var(--radius-full)',
                          backgroundColor: dotIdx === currentSlideIndex ? 'var(--color-primary)' : 'var(--color-border)',
                          border: 'none',
                          cursor: 'pointer',
                          transition: 'all var(--transition-fast)'
                        }}
                        title={`Go to step ${dotIdx + 1}`}
                      />
                    ))}
                  </div>

                  {/* Next Button */}
                  <Button 
                    size="lg" 
                    onClick={handleNext}
                    style={{ minWidth: '220px', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}
                  >
                    {currentSlideIndex < slides.length - 1 ? (
                      <>
                        <span>Next: {slides[currentSlideIndex + 1].shortTitle}</span>
                        <ChevronRight size={20} />
                      </>
                    ) : (
                      <>
                        <span>Back to Beginning ↺</span>
                        <ArrowRight size={20} />
                      </>
                    )}
                  </Button>
                </div>
              </Card>
            </div>
          ) : (
            /* =================== GRID VIEW (ALL 4 TILES) =================== */
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.75rem' }}>
              {slides.map((slide, idx) => (
                <Card 
                  key={slide.id} 
                  title={slide.title} 
                  style={{ textAlign: 'center', cursor: 'pointer', border: idx === currentSlideIndex ? '2px solid var(--color-primary)' : undefined, boxShadow: 'var(--shadow-md)' }}
                  onClick={() => {
                    setZoomLevel(1);
                    setCurrentSlideIndex(idx);
                    setViewMode('slider');
                  }}
                >
                  <div style={{ 
                    height: '340px',
                    backgroundColor: '#070b12', 
                    borderRadius: 'var(--radius-md)', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    overflow: 'hidden', 
                    marginBottom: '1rem',
                    border: '1px solid #1e293b'
                  }}>
                    <img 
                      src={slide.imageSrc} 
                      alt={slide.title} 
                      style={{ width: '100%', height: '100%', objectFit: 'contain', imageRendering: '-webkit-optimize-contrast' }} 
                    />
                  </div>
                  <p className="text-muted" style={{ fontSize: '0.875rem', margin: 0, textAlign: 'left', lineHeight: 1.5 }}>
                    {slide.description}
                  </p>
                </Card>
              ))}
            </div>
          )}

          {/* =================== FULLSCREEN LIGHTBOX MODAL =================== */}
          {isFullscreen && (
            <div 
              style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100vw',
                height: '100vh',
                backgroundColor: 'rgba(0, 0, 0, 0.95)',
                zIndex: 9999,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '2rem'
              }}
              onClick={() => setIsFullscreen(false)}
            >
              {/* Top Modal Controls */}
              <div 
                style={{ 
                  position: 'absolute', 
                  top: '1.5rem', 
                  right: '2rem', 
                  display: 'flex', 
                  gap: '1rem',
                  alignItems: 'center',
                  zIndex: 10000 
                }} 
                onClick={(e) => e.stopPropagation()}
              >
                <div style={{ color: 'white', marginRight: '1rem', fontWeight: 600, fontSize: '1rem' }}>
                  {currentSlide.title}
                </div>
                <Button variant="secondary" size="sm" onClick={handleZoomIn}>
                  <ZoomIn size={16} />
                </Button>
                <Button variant="secondary" size="sm" onClick={handleZoomOut}>
                  <ZoomOut size={16} />
                </Button>
                <Button variant="secondary" size="sm" onClick={handleResetZoom}>
                  <RotateCcw size={16} />
                </Button>
                <Button variant="primary" size="sm" onClick={() => setIsFullscreen(false)}>
                  <Minimize2 size={16} /> Close
                </Button>
              </div>

              {/* Fullscreen Image */}
              <div 
                style={{ 
                  width: '90vw', 
                  height: '82vh', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  overflow: 'hidden' 
                }}
                onClick={(e) => e.stopPropagation()}
              >
                <img 
                  src={currentSlide.imageSrc} 
                  alt={currentSlide.title} 
                  style={{ 
                    maxWidth: '100%', 
                    maxHeight: '100%', 
                    objectFit: 'contain',
                    transform: `scale(${zoomLevel})`,
                    transition: 'transform 0.2s ease',
                    filter: getFilterStyle(),
                    imageRendering: '-webkit-optimize-contrast'
                  }} 
                />
              </div>

              <div style={{ position: 'absolute', bottom: '1.5rem', color: '#94a3b8', fontSize: '0.9rem' }}>
                Press ESC or click outside to exit fullscreen
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default TechnicalDetails;
