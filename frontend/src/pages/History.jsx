function History() {
  return (
    <section>
      <div className="page-header mb-4">
        <div>
          <p className="text-primary fw-semibold mb-1">Records</p>
          <h1 className="h3 mb-1">History</h1>
          <p className="text-secondary mb-0">Review completed inspections once records are available.</p>
        </div>
      </div>

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
              <tr>
                <td colSpan="4" className="text-center text-secondary py-5">
                  No prediction history available
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

export default History;
