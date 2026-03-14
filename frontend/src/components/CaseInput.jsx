import { useRef } from 'react';

export default function CaseInput({
  samples,
  onSubmit,
  loading,
  onImageSelect,
  onSampleSelect,
  currentImage,
  currentQuestion,
  onQuestionChange,
}) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onImageSelect?.(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      onImageSelect?.(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleSampleChange = async (e) => {
    const sampleId = e.target.value;
    if (sampleId === '') return;

    const sample = samples.find(s => String(s.id) === sampleId);
    if (sample) {
      onSampleSelect?.(sample);
    }
  };

  const handleSubmit = () => {
    if (!currentImage || !currentQuestion?.trim()) return;
    onSubmit(currentImage, currentQuestion.trim());
  };

  return (
    <div className="panel-body">
      {/* Image Upload */}
      <div className="form-group">
        <label className="form-label">Medical Image</label>
        <div
          className={`upload-zone ${currentImage ? 'has-file' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
            id="image-upload-input"
          />
          {currentImage ? (
            <>
              <span className="upload-zone-icon">✓</span>
              <span className="upload-zone-filename">{currentImage.name}</span>
              <span className="upload-zone-text">Click to change</span>
            </>
          ) : (
            <>
              <span className="upload-zone-icon">🩻</span>
              <span className="upload-zone-text">
                Drop a medical image here<br />or click to browse
              </span>
            </>
          )}
        </div>
      </div>

      {/* Sample Case Selector */}
      <div className="form-group">
        <label className="form-label">Or select a sample case</label>
        <select
          className="form-select"
          onChange={handleSampleChange}
          defaultValue=""
          id="sample-case-select"
        >
          <option value="">— Select a sample —</option>
          {samples.map((s) => (
            <option key={s.id} value={s.id}>
              [{s.question_type?.replace(/_/g, ' ')}] {s.question.slice(0, 60)}
              {s.question.length > 60 ? '...' : ''}
            </option>
          ))}
        </select>
      </div>

      {/* Question Input */}
      <div className="form-group">
        <label className="form-label">Your Question</label>
        <textarea
          className="form-textarea"
          placeholder="e.g., What part of the body is shown in this image?"
          value={currentQuestion}
          onChange={(e) => onQuestionChange?.(e.target.value)}
          rows={3}
          id="question-input"
        />
      </div>

      {/* Submit Button */}
      <button
        className="btn btn-primary btn-full"
        onClick={handleSubmit}
        disabled={loading || !currentImage || !currentQuestion?.trim()}
        id="analyze-button"
      >
        {loading ? (
          <>
            <span className="loading-spinner" style={{ width: 16, height: 16, borderWidth: 2 }}></span>
            Analyzing...
          </>
        ) : (
          <>🔬 Analyze Case</>
        )}
      </button>

      {/* Disclaimer */}
      <div className="disclaimer-bar" style={{ marginTop: 'auto' }}>
        <span>⚠️</span>
        <span>For educational and research purposes only. Not for clinical use.</span>
      </div>
    </div>
  );
}
