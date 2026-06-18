import { NavLink } from 'react-router-dom';

const navItems = [
  { path: '/dashboard', label: 'Dashboard' },
  { path: '/inspection', label: 'Image Inspection' },
  { path: '/history', label: 'History' },
  { path: '/profile', label: 'Profile' },
];

function SidebarNav() {
  return (
    <ul className="nav nav-pills flex-column gap-2">
      {navItems.map((item) => (
        <li className="nav-item" key={item.path}>
          <NavLink
            to={item.path}
            className={({ isActive }) =>
              `nav-link sidebar-link ${isActive ? 'active' : 'text-dark'}`
            }
          >
            {item.label}
          </NavLink>
        </li>
      ))}
    </ul>
  );
}

function Sidebar() {
  return (
    <>
      <aside className="app-sidebar d-none d-lg-flex flex-column border-end bg-white">
        <div className="px-4 py-4 border-bottom">
          <div className="fw-bold">QualityVision AI</div>
          <div className="small text-secondary">Inspection Workspace</div>
        </div>
        <div className="p-3">
          <SidebarNav />
        </div>
      </aside>

      <div
        className="offcanvas offcanvas-start"
        tabIndex="-1"
        id="mobileSidebar"
        aria-labelledby="mobileSidebarLabel"
      >
        <div className="offcanvas-header border-bottom">
          <div>
            <h5 className="offcanvas-title" id="mobileSidebarLabel">
              QualityVision AI
            </h5>
            <div className="small text-secondary">Inspection Workspace</div>
          </div>
          <button
            type="button"
            className="btn-close"
            data-bs-dismiss="offcanvas"
            aria-label="Close"
          />
        </div>
        <div className="offcanvas-body">
          <SidebarNav />
        </div>
      </div>
    </>
  );
}

export default Sidebar;
