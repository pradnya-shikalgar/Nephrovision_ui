import React, { useState, useRef } from 'react';
import { UploadCloud } from 'lucide-react';
import './FileUpload.css';

const FileUpload = ({ onFileSelect }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selected = e.dataTransfer.files[0];
      setFile(selected);
      if (onFileSelect) onFileSelect(selected);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      if (onFileSelect) onFileSelect(selected);
    }
  };

  const triggerSelect = () => {
    fileInputRef.current.click();
  };

  return (
    <div 
      className={`file-upload-container ${isDragging ? 'dragging' : ''}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={triggerSelect}
    >
      <input 
        ref={fileInputRef}
        type="file" 
        className="file-upload-input" 
        onChange={handleChange}
        accept="image/*,.dcm"
      />
      
      <div className="file-upload-content">
        <UploadCloud size={48} className="upload-icon" />
        <h4 className="upload-title">
          {file ? file.name : "Drag & Drop CT Image"}
        </h4>
        <p className="upload-subtitle text-muted">
          {file ? `${(file.size / (1024*1024)).toFixed(2)} MB` : "or click to select file"}
        </p>
      </div>
    </div>
  );
};

export default FileUpload;
