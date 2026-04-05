import { useRef } from 'react';

export default function CaseInput({
  onSubmit,
  loading,
  onImageSelect,
  currentImage,
  currentQuestion,
  onQuestionChange,
  onClear,
  locked = false,
}) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    if (locked) return;
    const file = e.target.files?.[0];
    if (file) {
      onImageSelect?.(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (locked) return;
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      onImageSelect?.(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleSubmit = () => {
    if (!currentImage || !currentQuestion?.trim()) return;
    onSubmit(currentImage, currentQuestion.trim());
  };

  const handleClear = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    onClear?.();
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
          style={locked ? { opacity: 0.6, pointerEvents: 'none' } : undefined}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
            id="image-upload-input"
            disabled={locked}
          />
          {currentImage ? (
            <>
              <span className="upload-zone-icon">✓</span>
              <span className="upload-zone-filename">{currentImage.name}</span>
              <span className="upload-zone-text">Click to change</span>
            </>
          ) : (
            <>
              <span className="upload-zone-icon">🏥</span>
              <span className="upload-zone-text">
                Drop a medical image here<br />or click to browse
              </span>
            </>
          )}
          {locked && (
            <div className="split-guidance-text" style={{ marginTop: 'var(--space-2)' }}>
              Clear the current case before uploading another.
            </div>
          )}
        </div>
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

      {/* Actions */}
      <div className="button-row" style={{ display: 'flex', gap: 8 }}>
        <button
          className="btn btn-secondary"
          onClick={handleClear}
          disabled={loading || (!currentImage && !currentQuestion)}
          type="button"
          id="clear-button"
          style={{ flex: 1 }}
        >
          Clear
        </button>
        <button
          className="btn btn-primary"
          onClick={handleSubmit}
          disabled={loading || !currentImage || !currentQuestion?.trim()}
          id="analyze-button"
          style={{ flex: 2 }}
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
      </div>

      {/* Disclaimer */}
      <div className="disclaimer-bar" style={{ marginTop: 'auto' }}>
        <span>⚠️</span>
        <span>For educational and research purposes only. Not for clinical use.</span>
      </div>
    </div>
  );
}
