// Centralized API Base URL configuration
// Points to your live Render backend by default!
export const API_BASE = (
  import.meta.env.VITE_API_URL || 
  import.meta.env.BACKEND_API_URL || 
  'https://nephrovision-ui.onrender.com'
).replace(/\/+$/, '');
