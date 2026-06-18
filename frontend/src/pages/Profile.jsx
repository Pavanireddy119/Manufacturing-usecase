import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, clearAuth } from '../api/client';

function Profile() {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    api
      .me()
      .then((data) => {
        if (active) setUser(data);
      })
      .catch((err) => {
        if (active) setError(err.message || 'Failed to load profile.');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const handleLogout = () => {
    clearAuth();
    navigate('/');
  };

  const display = (value) => {
    if (loading) return 'Loading…';
    return value || 'No Data Available';
  };

  const memberSince = user?.created_at
    ? new Date(user.created_at).toLocaleDateString()
    : null;

  return (
    <section>
      <div className="page-header mb-4">
        <div>
          <p className="text-primary fw-semibold mb-1">Account</p>
          <h1 className="h3 mb-1">Profile</h1>
          <p className="text-secondary mb-0">User profile details</p>
        </div>
      </div>

      {error && (
        <div className="alert alert-danger" role="alert">
          {error}
        </div>
      )}

      <div className="bg-white border rounded-2 p-4">
        <div className="row g-3">
          <div className="col-12">
            <p className="text-secondary mb-1">Username:</p>
            <p className="fw-semibold mb-0">{display(user?.username)}</p>
          </div>
          <div className="col-12">
            <p className="text-secondary mb-1">Email:</p>
            <p className="fw-semibold mb-0">{display(user?.email)}</p>
          </div>
          <div className="col-12">
            <p className="text-secondary mb-1">Member Since:</p>
            <p className="fw-semibold mb-0">{display(memberSince)}</p>
          </div>
        </div>

        <div className="d-flex flex-column flex-sm-row gap-2 mt-4">
          <button className="btn btn-primary" type="button" onClick={() => navigate('/profile/edit')}>
            Edit Profile
          </button>
          <button
            className="btn btn-outline-primary"
            type="button"
            onClick={() => navigate('/profile/change-password')}
          >
            Change Password
          </button>
          <button className="btn btn-outline-danger ms-sm-auto" type="button" onClick={handleLogout}>
            Logout
          </button>
        </div>
      </div>
    </section>
  );
}

export default Profile;
