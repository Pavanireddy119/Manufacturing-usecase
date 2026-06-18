// Lightweight API client for the QualityVision AI backend.
// Base URL is configurable via VITE_API_BASE_URL (see .env.example),
// defaulting to the FastAPI dev server.

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

const TOKEN_KEY = 'qvai_token';

export const getToken = () => localStorage.getItem(TOKEN_KEY);
export const setToken = (token) => localStorage.setItem(TOKEN_KEY, token);
export const clearAuth = () => {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem('isAuthenticated');
};
export const isAuthenticated = () => Boolean(getToken());

async function request(path, { method = 'GET', body, auth = true, isForm = false } = {}) {
  const headers = {};
  if (auth) {
    const token = getToken();
    if (token) headers.Authorization = `Bearer ${token}`;
  }
  let payload = body;
  if (body && !isForm) {
    headers['Content-Type'] = 'application/json';
    payload = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { method, headers, body: payload });
  } catch {
    throw new Error('Cannot reach the server. Is the backend running?');
  }

  let data = null;
  const text = await response.text();
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }

  if (!response.ok) {
    const message = (data && (data.message || data.detail)) || `Request failed (${response.status})`;
    throw new Error(message);
  }
  return data;
}

export const api = {
  signup: (username, email, password) =>
    request('/api/auth/signup', { method: 'POST', auth: false, body: { username, email, password } }),

  login: (username, password) =>
    request('/api/auth/login', { method: 'POST', auth: false, body: { username, password } }),

  uploadImage: (file) => {
    const form = new FormData();
    form.append('file', file);
    return request('/api/upload', { method: 'POST', body: form, isForm: true });
  },

  predict: (imageId) => request(`/api/predict/${imageId}`, { method: 'POST' }),

  history: () => request('/api/history'),
};
