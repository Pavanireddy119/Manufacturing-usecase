import { useNavigate } from 'react-router-dom';

function Navbar() {
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem('isAuthenticated');
    navigate('/');
  };

  return (
    <nav className="navbar navbar-expand bg-white border-bottom sticky-top app-navbar">
      <div className="container-fluid">
        <button
          className="btn btn-outline-secondary d-lg-none me-2"
          type="button"
          data-bs-toggle="offcanvas"
          data-bs-target="#mobileSidebar"
          aria-controls="mobileSidebar"
          aria-label="Open navigation"
        >
          <span className="navbar-toggler-icon" />
        </button>

        <span className="navbar-brand mb-0 h1">IntelliInspect</span>

        <div className="ms-auto d-flex align-items-center gap-2">
          <button className="btn btn-primary btn-sm" type="button" onClick={handleLogout}>
            Logout
          </button>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
