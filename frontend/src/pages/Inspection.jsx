import { useEffect, useState } from 'react';
import { api } from '../api/client';

function Inspection() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    // Revoke the object URL when it changes or the component unmounts.
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleFileChange = (event) => {
    const selected = event.target.files?.[0] || null;
    setFile(selected);
    setResult(null);
    setError('');
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(selected ? URL.createObjectURL(selected) : '');
  };

  const handlePredict = async () => {
    if (!file) {
      setError('Please choose an image first.');
      return;
    }
    setLoading(true);
    setError('');
    setResult(null);
    try {
      // Upload the image, then run prediction against the stored image id.
      const uploaded = await api.uploadImage(file);
      const prediction = await api.predict(uploaded.image_id);
      setResult(prediction);
    } catch (err) {
      setError(err.message || 'Prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  const isDefective = result?.prediction === 'Defective';

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
            <input
              className="form-control"
              id="inspectionImage"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
            />

            <div className="preview-box mt-4">
              {previewUrl ? (
                <img src={previewUrl} alt="Selected preview" style={{ maxWidth: '100%', maxHeight: '240px' }} />
              ) : (
                <span>Image Preview Area</span>
              )}
            </div>

            {error && (
              <div className="alert alert-danger py-2 mt-3 mb-0" role="alert">
                {error}
              </div>
            )}

            <button
              className="btn btn-primary mt-4 w-100"
              type="button"
              onClick={handlePredict}
              disabled={loading || !file}
            >
              {loading ? 'Analyzing…' : 'Predict'}
            </button>
          </div>
        </div>

        <div className="col-lg-6">
          <div className="bg-white border rounded-2 p-4 h-100">
            <h2 className="h5 mb-3">Result Area</h2>
            <div className="result-list mb-4">
              <div className="d-flex justify-content-between py-2 border-bottom">
                <span className="text-secondary">Prediction:</span>
                <span className={`fw-semibold ${result ? (isDefective ? 'text-danger' : 'text-success') : ''}`}>
                  {result ? result.prediction : 'No Data Available'}
                </span>
              </div>
              <div className="d-flex justify-content-between py-2 border-bottom">
                <span className="text-secondary">Confidence Score:</span>
                <span className="fw-semibold">
                  {result ? `${Number(result.confidence_score).toFixed(2)}%` : 'No Data Available'}
                </span>
              </div>
              <div className="d-flex justify-content-between py-2 border-bottom">
                <span className="text-secondary">Status:</span>
                <span className="fw-semibold">
                  {result ? (isDefective ? 'Inspection Failed' : 'Inspection Passed') : 'No Data Available'}
                </span>
              </div>
              <div className="d-flex justify-content-between py-2 border-bottom">
                <span className="text-secondary">Damage Type:</span>
                <span className="fw-semibold">{result?.damage_type || '—'}</span>
              </div>
              <div className="d-flex justify-content-between py-2 border-bottom">
                <span className="text-secondary">Location:</span>
                <span className="fw-semibold">{result?.damage_location || '—'}</span>
              </div>
              <div className="d-flex justify-content-between py-2 border-bottom">
                <span className="text-secondary">Severity:</span>
                <span className="fw-semibold">{result?.severity || '—'}</span>
              </div>
              <div className="d-flex justify-content-between py-2">
                <span className="text-secondary">Recommendation:</span>
                <span className="fw-semibold">{result?.recommendation || '—'}</span>
              </div>
            </div>

            <div className="preview-box preview-box-sm">
              {previewUrl ? (
                <img src={previewUrl} alt="Inspected" style={{ maxWidth: '100%', maxHeight: '160px' }} />
              ) : (
                <span>Annotated Image Placeholder</span>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default Inspection;
