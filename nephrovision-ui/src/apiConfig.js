// Centralized API Base URL configuration
// Automatically targets local FastAPI on localhost, and live Render backend on production!
export const API_BASE = (
  import.meta.env.VITE_API_URL || 
  import.meta.env.BACKEND_API_URL || 
  (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'http://localhost:8000'
    : 'https://nephrovision-ui.onrender.com')
).replace(/\/+$/, '');

