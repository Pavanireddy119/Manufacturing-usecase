import { Link } from 'react-router-dom';

const analyticsCards = ['Total Inspection', 'Defective Count', 'Non-Defective Count'];

const quickAccessCards = [
  { title: 'Image Inspection', text: 'Upload and inspect manufacturing images.', path: '/inspection' },
  { title: 'History', text: 'Review saved prediction history.', path: '/history' },
  { title: 'Profile', text: 'Manage account information.', path: '/profile' },
];

function Dashboard() {
  return (
    <section>
      <div className="page-header mb-4">
        <div>
          <p className="text-primary fw-semibold mb-1">Overview</p>
          <h1 className="h3 mb-1">Dashboard</h1>
          <p className="text-secondary mb-0">Monitor inspection readiness and quality workflows.</p>
        </div>
      </div>

      <div className="row g-3 mb-4">
        {analyticsCards.map((title) => (
          <div className="col-md-4" key={title}>
            <article className="dashboard-card bg-white border rounded-2 p-4 h-100">
              <p className="text-secondary mb-2">{title}</p>
              <h2 className="h5 mb-0">No Data Available</h2>
            </article>
          </div>
        ))}
      </div>

      <div className="row g-3 mb-4">
        <div className="col-lg-7">
          <div className="bg-white border rounded-2 p-4 h-100">
            <h2 className="h5 mb-3">Inspection Trend</h2>
            <div className="chart-placeholder">
              <span>No Data Available</span>
            </div>
          </div>
        </div>
        <div className="col-lg-5">
          <div className="bg-white border rounded-2 p-4 h-100">
            <h2 className="h5 mb-3">Defect Distribution</h2>
            <div className="chart-placeholder chart-placeholder-sm">
              <span>No Data Available</span>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3">
        {quickAccessCards.map((card) => (
          <div className="col-md-4" key={card.path}>
            <Link to={card.path} className="text-decoration-none">
              <article className="dashboard-card bg-white border rounded-2 p-4 h-100">
                <h2 className="h5 mb-2">{card.title}</h2>
                <p className="text-secondary mb-0">{card.text}</p>
              </article>
            </Link>
          </div>
        ))}
      </div>
    </section>
  );
}

export default Dashboard;
