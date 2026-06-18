import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api/client';

const quickAccessCards = [
  { title: 'Image Inspection', text: 'Upload and inspect manufacturing images.', path: '/inspection' },
  { title: 'History', text: 'Review saved prediction history.', path: '/history' },
  { title: 'Profile', text: 'Manage account information.', path: '/profile' },
];

function Dashboard() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    api
      .history()
      .then((data) => {
        if (active) setRecords(Array.isArray(data) ? data : []);
      })
      .catch((err) => {
        if (active) setError(err.message || 'Failed to load dashboard data.');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const stats = useMemo(() => {
    const total = records.length;
    const defective = records.filter((r) => r.prediction === 'Defective').length;
    const nonDefective = total - defective;
    const defectiveRate = total ? Math.round((defective / total) * 100) : 0;
    return { total, defective, nonDefective, defectiveRate };
  }, [records]);

  const recent = useMemo(() => records.slice(0, 5), [records]);

  const analyticsCards = [
    { title: 'Total Inspection', value: stats.total },
    { title: 'Defective Count', value: stats.defective },
    { title: 'Non-Defective Count', value: stats.nonDefective },
  ];

  const renderValue = (value) => {
    if (loading) return '…';
    return value;
  };

  return (
    <section>
      <div className="page-header mb-4">
        <div>
          <p className="text-primary fw-semibold mb-1">Overview</p>
          <h1 className="h3 mb-1">Dashboard</h1>
          <p className="text-secondary mb-0">Monitor inspection readiness and quality workflows.</p>
        </div>
      </div>

      {error && (
        <div className="alert alert-danger" role="alert">
          {error}
        </div>
      )}

      <div className="row g-3 mb-4">
        {analyticsCards.map((card) => (
          <div className="col-md-4" key={card.title}>
            <article className="dashboard-card bg-white border rounded-2 p-4 h-100">
              <p className="text-secondary mb-2">{card.title}</p>
              <h2 className="h5 mb-0">{renderValue(card.value)}</h2>
            </article>
          </div>
        ))}
      </div>

      <div className="row g-3 mb-4">
        <div className="col-lg-7">
          <div className="bg-white border rounded-2 p-4 h-100">
            <h2 className="h5 mb-3">Recent Inspections</h2>
            {loading ? (
              <p className="text-secondary mb-0">Loading…</p>
            ) : recent.length === 0 ? (
              <p className="text-secondary mb-0">No inspections yet. Run one from the Inspection page.</p>
            ) : (
              <ul className="list-group list-group-flush">
                {recent.map((r) => (
                  <li
                    key={r.id}
                    className="list-group-item d-flex justify-content-between align-items-center px-0"
                  >
                    <span className="text-truncate me-3" style={{ maxWidth: '60%' }}>
                      {r.image_name}
                    </span>
                    <span
                      className={`badge ${r.prediction === 'Defective' ? 'bg-danger' : 'bg-success'}`}
                    >
                      {r.prediction} · {Number(r.confidence_score).toFixed(0)}%
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
        <div className="col-lg-5">
          <div className="bg-white border rounded-2 p-4 h-100">
            <h2 className="h5 mb-3">Defect Distribution</h2>
            {loading ? (
              <p className="text-secondary mb-0">Loading…</p>
            ) : stats.total === 0 ? (
              <p className="text-secondary mb-0">No data yet.</p>
            ) : (
              <>
                <div className="d-flex justify-content-between mb-1">
                  <span className="text-secondary">Defective</span>
                  <span className="fw-semibold">{stats.defectiveRate}%</span>
                </div>
                <div className="progress mb-3" style={{ height: '14px' }}>
                  <div
                    className="progress-bar bg-danger"
                    role="progressbar"
                    style={{ width: `${stats.defectiveRate}%` }}
                    aria-valuenow={stats.defectiveRate}
                    aria-valuemin="0"
                    aria-valuemax="100"
                  />
                </div>
                <div className="d-flex justify-content-between mb-1">
                  <span className="text-secondary">Non-Defective</span>
                  <span className="fw-semibold">{100 - stats.defectiveRate}%</span>
                </div>
                <div className="progress" style={{ height: '14px' }}>
                  <div
                    className="progress-bar bg-success"
                    role="progressbar"
                    style={{ width: `${100 - stats.defectiveRate}%` }}
                    aria-valuenow={100 - stats.defectiveRate}
                    aria-valuemin="0"
                    aria-valuemax="100"
                  />
                </div>
              </>
            )}
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
