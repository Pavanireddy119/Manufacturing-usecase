import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

function Register() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
  });
  const [errors, setErrors] = useState({});

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData((current) => ({ ...current, [name]: value }));
    setErrors((current) => ({ ...current, [name]: '' }));
  };

  const validateForm = () => {
    const nextErrors = {};
    const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (!formData.username.trim()) {
      nextErrors.username = 'Username is required.';
    }

    if (!formData.email.trim()) {
      nextErrors.email = 'Email is required.';
    } else if (!emailPattern.test(formData.email)) {
      nextErrors.email = 'Enter a valid email address.';
    }

    if (!formData.password) {
      nextErrors.password = 'Password is required.';
    } else if (formData.password.length < 6) {
      nextErrors.password = 'Password must be at least 6 characters.';
    }

    if (!formData.confirmPassword) {
      nextErrors.confirmPassword = 'Confirm your password.';
    } else if (formData.confirmPassword !== formData.password) {
      nextErrors.confirmPassword = 'Passwords do not match.';
    }

    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleSubmit = (event) => {
    event.preventDefault();

    if (!validateForm()) {
      return;
    }

    navigate('/');
  };

  return (
    <main className="auth-page">
      <section className="auth-panel shadow-sm">
        <div className="mb-4">
          <p className="text-primary fw-semibold mb-1">QualityVision AI</p>
          <h1 className="h3 mb-2">Create Account</h1>
          <p className="text-secondary mb-0">Set up access to the inspection workspace.</p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="mb-3">
            <label className="form-label" htmlFor="registerUsername">
              Username
            </label>
            <input
              className={`form-control ${errors.username ? 'is-invalid' : ''}`}
              id="registerUsername"
              name="username"
              type="text"
              placeholder="Choose username"
              value={formData.username}
              onChange={handleChange}
            />
            {errors.username && <div className="invalid-feedback">{errors.username}</div>}
          </div>
          <div className="mb-3">
            <label className="form-label" htmlFor="registerEmail">
              Email
            </label>
            <input
              className={`form-control ${errors.email ? 'is-invalid' : ''}`}
              id="registerEmail"
              name="email"
              type="email"
              placeholder="name@company.com"
              value={formData.email}
              onChange={handleChange}
            />
            {errors.email && <div className="invalid-feedback">{errors.email}</div>}
          </div>
          <div className="mb-3">
            <label className="form-label" htmlFor="registerPassword">
              Password
            </label>
            <input
              className={`form-control ${errors.password ? 'is-invalid' : ''}`}
              id="registerPassword"
              name="password"
              type="password"
              placeholder="Create password"
              value={formData.password}
              onChange={handleChange}
            />
            {errors.password && <div className="invalid-feedback">{errors.password}</div>}
          </div>
          <div className="mb-4">
            <label className="form-label" htmlFor="confirmPassword">
              Confirm Password
            </label>
            <input
              className={`form-control ${errors.confirmPassword ? 'is-invalid' : ''}`}
              id="confirmPassword"
              name="confirmPassword"
              type="password"
              placeholder="Confirm password"
              value={formData.confirmPassword}
              onChange={handleChange}
            />
            {errors.confirmPassword && (
              <div className="invalid-feedback">{errors.confirmPassword}</div>
            )}
          </div>
          <button className="btn btn-primary w-100" type="submit">
            Register
          </button>
        </form>

        <p className="text-center text-secondary mt-4 mb-0">
          Already have an account?{' '}
          <Link to="/" className="text-decoration-none">
            Login
          </Link>
        </p>
      </section>
    </main>
  );
}

export default Register;
