import { useEffect, useState } from 'react';
import { api } from '../api/client';

function History() {
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
        if (active) setError(err.message || 'Failed to load history.');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const formatTime = (value) => {
    if (!value) return '—';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
  };

  return (
    <section>
      <div className="page-header mb-4">
        <div>
          <p className="text-primary fw-semibold mb-1">Records</p>
          <h1 className="h3 mb-1">History</h1>
          <p className="text-secondary mb-0">Review completed inspections once records are available.</p>
        </div>
      </div>

      {error && (
        <div className="alert alert-danger" role="alert">
          {error}
        </div>
      )}

      <div className="bg-white border rounded-2 p-3 p-md-4">
        <div className="table-responsive">
          <table className="table align-middle mb-0">
            <thead>
              <tr>
                <th scope="col">Image Name</th>
                <th scope="col">Prediction</th>
                <th scope="col">Confidence Score</th>
                <th scope="col">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan="4" className="text-center text-secondary py-5">
                    Loading…
                  </td>
                </tr>
              ) : records.length === 0 ? (
                <tr>
                  <td colSpan="4" className="text-center text-secondary py-5">
                    No prediction history available
                  </td>
                </tr>
              ) : (
                records.map((record) => (
                  <tr key={record.id}>
                    <td>{record.image_name}</td>
                    <td>
                      <span
                        className={`fw-semibold ${
                          record.prediction === 'Defective' ? 'text-danger' : 'text-success'
                        }`}
                      >
                        {record.prediction}
                      </span>
                    </td>
                    <td>{Number(record.confidence_score).toFixed(2)}%</td>
                    <td>{formatTime(record.created_at)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

export default History;
