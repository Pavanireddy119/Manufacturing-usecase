import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function ChangePassword() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    currentPassword: '',
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

    if (!formData.currentPassword) {
      nextErrors.currentPassword = 'Current password is required.';
    }

    if (!formData.newPassword) {
      nextErrors.newPassword = 'New password is required.';
    } else if (formData.newPassword.length < 6) {
      nextErrors.newPassword = 'Password must be at least 6 characters.';
    } else if (formData.newPassword === formData.currentPassword) {
      nextErrors.newPassword = 'New password must be different from current password.';
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

    navigate('/profile');
  };

  return (
    <section>
      <div className="page-header mb-4">
        <div>
          <p className="text-primary fw-semibold mb-1">Account</p>
          <h1 className="h3 mb-1">Change Password</h1>
          <p className="text-secondary mb-0">Update your account password.</p>
        </div>
      </div>

      <div className="bg-white border rounded-2 p-4">
        <form onSubmit={handleSubmit} noValidate>
          <div className="row g-3">
            <div className="col-md-6">
              <label className="form-label" htmlFor="currentPassword">
                Current Password
              </label>
              <input
                className={`form-control ${errors.currentPassword ? 'is-invalid' : ''}`}
                id="currentPassword"
                name="currentPassword"
                type="password"
                placeholder="Enter current password"
                value={formData.currentPassword}
                onChange={handleChange}
              />
              {errors.currentPassword && (
                <div className="invalid-feedback">{errors.currentPassword}</div>
              )}
            </div>

            <div className="col-md-6">
              <label className="form-label" htmlFor="profileNewPassword">
                New Password
              </label>
              <input
                className={`form-control ${errors.newPassword ? 'is-invalid' : ''}`}
                id="profileNewPassword"
                name="newPassword"
                type="password"
                placeholder="Enter new password"
                value={formData.newPassword}
                onChange={handleChange}
              />
              {errors.newPassword && <div className="invalid-feedback">{errors.newPassword}</div>}
            </div>

            <div className="col-md-6">
              <label className="form-label" htmlFor="profileConfirmPassword">
                Confirm Password
              </label>
              <input
                className={`form-control ${errors.confirmPassword ? 'is-invalid' : ''}`}
                id="profileConfirmPassword"
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
          </div>

          <div className="d-flex flex-column flex-sm-row gap-2 mt-4">
            <button className="btn btn-primary" type="submit">
              Change Password
            </button>
            <button
              className="btn btn-outline-secondary"
              type="button"
              onClick={() => navigate('/profile')}
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </section>
  );
}

export default ChangePassword;
