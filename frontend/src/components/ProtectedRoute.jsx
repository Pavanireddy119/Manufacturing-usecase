import { Navigate, Outlet } from 'react-router-dom';
import { isAuthenticated as hasToken } from '../api/client';

function ProtectedRoute() {
  if (!hasToken()) {
    return <Navigate to="/" replace />;
  }

  return <Outlet />;
}

export default ProtectedRoute;
