import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

function Login() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    username: '',
    password: '',
  });
  const [errors, setErrors] = useState({});

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData((current) => ({ ...current, [name]: value }));
    setErrors((current) => ({ ...current, [name]: '' }));
  };

  const validateForm = () => {
    const nextErrors = {};

    if (!formData.username.trim()) {
      nextErrors.username = 'Username is required.';
    }

    if (!formData.password) {
      nextErrors.password = 'Password is required.';
    }

    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleSubmit = (event) => {
    event.preventDefault();

    if (!validateForm()) {
      return;
    }

    localStorage.setItem('isAuthenticated', 'true');
    navigate('/dashboard');
  };

  return (
    <main className="auth-page">
      <section className="auth-panel shadow-sm">
        <div className="mb-4">
          <p className="text-primary fw-semibold mb-1">QualityVision AI</p>
          <h1 className="h3 mb-2">Sign in</h1>
          <p className="text-secondary mb-0">Access the image inspection workspace.</p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="mb-3">
            <label className="form-label" htmlFor="username">
              Username
            </label>
            <input
              className={`form-control ${errors.username ? 'is-invalid' : ''}`}
              id="username"
              name="username"
              type="text"
              placeholder="Enter username"
              value={formData.username}
              onChange={handleChange}
            />
            {errors.username && <div className="invalid-feedback">{errors.username}</div>}
          </div>
          <div className="mb-3">
            <label className="form-label" htmlFor="password">
              Password
            </label>
            <input
              className={`form-control ${errors.password ? 'is-invalid' : ''}`}
              id="password"
              name="password"
              type="password"
              placeholder="Enter password"
              value={formData.password}
              onChange={handleChange}
            />
            {errors.password && <div className="invalid-feedback">{errors.password}</div>}
          </div>
          <div className="d-flex justify-content-between align-items-center mb-4">
            <div className="form-check">
              <input className="form-check-input" type="checkbox" id="remember" />
              <label className="form-check-label" htmlFor="remember">
                Remember me
              </label>
            </div>
            <Link to="/forgot-password" className="text-decoration-none">
              Forgot password?
            </Link>
          </div>
          <button className="btn btn-primary w-100" type="submit">
            Login
          </button>
        </form>

        <p className="text-center text-secondary mt-4 mb-0">
          New here?{' '}
          <Link to="/register" className="text-decoration-none">
            Create Account
          </Link>
        </p>
      </section>
    </main>
  );
}

export default Login;
