import { Outlet } from 'react-router-dom';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';

function MainLayout() {
  return (
    <div className="app-shell">
      <Sidebar />
      <div className="app-content">
        <Navbar />
        <main className="container-fluid px-3 px-md-4 py-4">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export default MainLayout;
