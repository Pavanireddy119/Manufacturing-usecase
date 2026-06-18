import { useNavigate } from 'react-router-dom';

function Profile() {
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem('isAuthenticated');
    navigate('/');
  };

  return (
    <section>
      <div className="page-header mb-4">
        <div>
          <p className="text-primary fw-semibold mb-1">Account</p>
          <h1 className="h3 mb-1">Profile</h1>
          <p className="text-secondary mb-0">User profile details</p>
        </div>
      </div>

      <div className="bg-white border rounded-2 p-4">
        <div className="row g-3">
          <div className="col-12">
            <p className="text-secondary mb-1">Username:</p>
            <p className="fw-semibold mb-0">No Data Available</p>
          </div>
          <div className="col-12">
            <p className="text-secondary mb-1">Email:</p>
            <p className="fw-semibold mb-0">No Data Available</p>
          </div>
          <div className="col-12">
            <p className="text-secondary mb-1">Role:</p>
            <p className="fw-semibold mb-0">No Data Available</p>
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
