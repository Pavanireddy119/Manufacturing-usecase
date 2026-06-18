function Inspection() {
  return (
    <section>
      <div className="page-header mb-4">
        <div>
          <p className="text-primary fw-semibold mb-1">Image Inspection</p>
          <h1 className="h3 mb-1">Inspection</h1>
          <p className="text-secondary mb-0">Prepare visual quality checks for manufacturing images.</p>
        </div>
      </div>

      <div className="row g-4">
        <div className="col-lg-6">
          <div className="bg-white border rounded-2 p-4 h-100">
            <h2 className="h5 mb-3">Upload Image</h2>
            <input className="form-control" id="inspectionImage" type="file" accept="image/*" />

            <div className="preview-box mt-4">
              <span>Image Preview Area</span>
            </div>

            <button className="btn btn-primary mt-4 w-100" type="button">
              Predict
            </button>
          </div>
        </div>

        <div className="col-lg-6">
          <div className="bg-white border rounded-2 p-4 h-100">
            <h2 className="h5 mb-3">Result Area</h2>
            <div className="result-list mb-4">
              <div className="d-flex justify-content-between py-2 border-bottom">
                <span className="text-secondary">Prediction:</span>
                <span className="fw-semibold">No Data Available</span>
              </div>
              <div className="d-flex justify-content-between py-2 border-bottom">
                <span className="text-secondary">Confidence Score:</span>
                <span className="fw-semibold">No Data Available</span>
              </div>
              <div className="d-flex justify-content-between py-2">
                <span className="text-secondary">Status:</span>
                <span className="fw-semibold">No Data Available</span>
              </div>
            </div>

            <div className="preview-box preview-box-sm">
              <span>Annotated Image Placeholder</span>
            </div>

            <button className="btn btn-outline-primary mt-4 w-100" type="button">
              Save Result
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}

export default Inspection;
