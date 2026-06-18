import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

function ForgotPassword() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    identifier: '',
    newPassword: '',
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

    if (!formData.identifier.trim()) {
      nextErrors.identifier = 'Email or username is required.';
    }

    if (!formData.newPassword) {
      nextErrors.newPassword = 'New password is required.';
    } else if (formData.newPassword.length < 6) {
      nextErrors.newPassword = 'Password must be at least 6 characters.';
    }

    if (!formData.confirmPassword) {
      nextErrors.confirmPassword = 'Confirm your new password.';
    } else if (formData.confirmPassword !== formData.newPassword) {
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
          <h1 className="h3 mb-2">Forgot Password</h1>
          <p className="text-secondary mb-0">Enter your email address to continue.</p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="mb-3">
            <label className="form-label" htmlFor="identifier">
              Email or Username
            </label>
            <input
              className={`form-control ${errors.identifier ? 'is-invalid' : ''}`}
              id="identifier"
              name="identifier"
              type="text"
              placeholder="Enter email or username"
              value={formData.identifier}
              onChange={handleChange}
            />
            {errors.identifier && <div className="invalid-feedback">{errors.identifier}</div>}
          </div>
          <div className="mb-3">
            <label className="form-label" htmlFor="newPassword">
              New Password
            </label>
            <input
              className={`form-control ${errors.newPassword ? 'is-invalid' : ''}`}
              id="newPassword"
              name="newPassword"
              type="password"
              placeholder="Enter new password"
              value={formData.newPassword}
              onChange={handleChange}
            />
            {errors.newPassword && <div className="invalid-feedback">{errors.newPassword}</div>}
          </div>
          <div className="mb-4">
            <label className="form-label" htmlFor="resetConfirmPassword">
              Confirm Password
            </label>
            <input
              className={`form-control ${errors.confirmPassword ? 'is-invalid' : ''}`}
              id="resetConfirmPassword"
              name="confirmPassword"
              type="password"
              placeholder="Confirm new password"
              value={formData.confirmPassword}
              onChange={handleChange}
            />
            {errors.confirmPassword && (
              <div className="invalid-feedback">{errors.confirmPassword}</div>
            )}
          </div>
          <button className="btn btn-primary w-100" type="submit">
            Reset Password
          </button>
        </form>

        <p className="text-center text-secondary mt-4 mb-0">
          <Link to="/" className="text-decoration-none">
            Back to login
          </Link>
        </p>
      </section>
    </main>
  );
}

export default ForgotPassword;
