// Centralized API Base URL configuration
// In production (Vercel), uses VITE_API_URL environment variable if set.
// Otherwise falls back to localhost:8000 for local development.
export const API_BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '');
