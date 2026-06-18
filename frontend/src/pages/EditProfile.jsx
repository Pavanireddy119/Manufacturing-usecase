import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function EditProfile() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    role: '',
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

    if (!formData.role.trim()) {
      nextErrors.role = 'Role is required.';
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
          <h1 className="h3 mb-1">Edit Profile</h1>
          <p className="text-secondary mb-0">Update your profile information.</p>
        </div>
      </div>

      <div className="bg-white border rounded-2 p-4">
        <form onSubmit={handleSubmit} noValidate>
          <div className="row g-3">
            <div className="col-md-6">
              <label className="form-label" htmlFor="editUsername">
                Username
              </label>
              <input
                className={`form-control ${errors.username ? 'is-invalid' : ''}`}
                id="editUsername"
                name="username"
                type="text"
                placeholder="Enter username"
                value={formData.username}
                onChange={handleChange}
              />
              {errors.username && <div className="invalid-feedback">{errors.username}</div>}
            </div>

            <div className="col-md-6">
              <label className="form-label" htmlFor="editEmail">
                Email
              </label>
              <input
                className={`form-control ${errors.email ? 'is-invalid' : ''}`}
                id="editEmail"
                name="email"
                type="email"
                placeholder="name@company.com"
                value={formData.email}
                onChange={handleChange}
              />
              {errors.email && <div className="invalid-feedback">{errors.email}</div>}
            </div>

            <div className="col-md-6">
              <label className="form-label" htmlFor="editRole">
                Role
              </label>
              <input
                className={`form-control ${errors.role ? 'is-invalid' : ''}`}
                id="editRole"
                name="role"
                type="text"
                placeholder="Enter role"
                value={formData.role}
                onChange={handleChange}
              />
              {errors.role && <div className="invalid-feedback">{errors.role}</div>}
            </div>
          </div>

          <div className="d-flex flex-column flex-sm-row gap-2 mt-4">
            <button className="btn btn-primary" type="submit">
              Save Changes
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

export default EditProfile;
