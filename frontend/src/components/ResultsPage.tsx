import type { InferenceResponse } from '../lib/api';

type Props = {
  inference: InferenceResponse | null;
  onBackToRoi: () => void;
};

function ResultCard({ title, url }: { title: string; url?: string }) {
  return (
    <div className="result-card">
      <div className="result-card-title">{title}</div>
      {url ? <img src={url} alt={title} /> : <div className="empty-panel">No image.</div>}
    </div>
  );
}

function WaterDetectionLegend() {
  return (
    <aside className="detection-legend" aria-label="H2O detection color legend">
      <div className="legend-title">Water score</div>
      <div className="legend-scale-row">
        <div className="legend-ramp" aria-hidden="true" />
        <div className="legend-labels">
          <span>High</span>
          <span>Mid</span>
          <span>Low</span>
        </div>
      </div>
      <div className="legend-swatches">
        <div><span className="legend-swatch high" /> stronger</div>
        <div><span className="legend-swatch mid" /> moderate</div>
        <div><span className="legend-swatch low" /> weaker</div>
        <div><span className="legend-swatch outside" /> outside search</div>
      </div>
      <div className="legend-note">Relative color scale for the displayed ROI.</div>
    </aside>
  );
}

function WaterDetectionCard({ url }: { url?: string }) {
  return (
    <div className="result-card detection-card">
      <div className="result-card-title">H2O Detection Overlay</div>
      <div className="detection-card-layout">
        {url ? <img src={url} alt="H2O Detection Overlay" /> : <div className="empty-panel">No image.</div>}
        <WaterDetectionLegend />
      </div>
    </div>
  );
}

function GroundTruthLegend() {
  return (
    <aside className="detection-legend truth-legend" aria-label="Ground truth label color legend">
      <div className="legend-title">Ground truth</div>
      <div className="legend-swatches truth-swatches">
        <div><span className="legend-swatch truth-wet" /> wet / H2O</div>
        <div><span className="legend-swatch truth-dry" /> dry</div>
        <div><span className="legend-swatch outside" /> ignored / outside</div>
      </div>
    </aside>
  );
}

function GroundTruthCard({ url, error, fallbackUrl }: {
  url?: string;
  error?: string | null;
  fallbackUrl?: string;
}) {
  if (!url) {
    return (
      <div className="result-card">
        <div className="result-card-title">Pseudo Color</div>
        {fallbackUrl ? <img src={fallbackUrl} alt="Pseudo Color" /> : <div className="empty-panel">No image.</div>}
        {error ? <div className="error-hint">{error}</div> : null}
      </div>
    );
  }

  return (
    <div className="result-card detection-card">
      <div className="result-card-title">Ground Truth Label</div>
      <div className="detection-card-layout">
        <img src={url} alt="Ground Truth Label" />
        <GroundTruthLegend />
      </div>
      {error ? <div className="error-hint">{error}</div> : null}
    </div>
  );
}

function formatMetric(value: unknown): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return '-';
  if (Math.abs(value) >= 1000) return value.toLocaleString();
  return value.toFixed(4);
}

function EvaluationPanel({ metrics, error, url }: {
  metrics?: Record<string, any> | null;
  error?: string | null;
  url?: string;
}) {
  if (!metrics && !error) return null;

  const confusion = metrics?.confusion_matrix;
  const tn = Array.isArray(confusion) ? confusion?.[0]?.[0] : undefined;
  const fp = Array.isArray(confusion) ? confusion?.[0]?.[1] : undefined;
  const fn = Array.isArray(confusion) ? confusion?.[1]?.[0] : undefined;
  const tp = Array.isArray(confusion) ? confusion?.[1]?.[1] : undefined;

  return (
    <div className="evaluation-panel">
      <div className="result-card-title">Evaluation Metrics</div>
      {error ? <div className="error-hint">{error}</div> : null}
      {metrics ? (
        <>
          <div className="metric-grid">
            <div><strong>F1</strong><span>{formatMetric(metrics.f1)}</span></div>
            <div><strong>Precision</strong><span>{formatMetric(metrics.precision)}</span></div>
            <div><strong>Recall</strong><span>{formatMetric(metrics.recall)}</span></div>
            <div><strong>IoU</strong><span>{formatMetric(metrics.iou)}</span></div>
            <div><strong>Accuracy</strong><span>{formatMetric(metrics.accuracy)}</span></div>
            <div><strong>Specificity</strong><span>{formatMetric(metrics.specificity)}</span></div>
            <div><strong>ROC-AUC</strong><span>{formatMetric(metrics.roc_auc)}</span></div>
            <div><strong>PR-AUC</strong><span>{formatMetric(metrics.pr_auc)}</span></div>
          </div>

          <div className="metric-meta">
            <div><strong>Threshold:</strong> {formatMetric(metrics.threshold)}</div>
            <div><strong>Valid pixels:</strong> {formatMetric(metrics.valid_pixels)}</div>
            <div><strong>Label wet rate:</strong> {formatMetric(metrics.label_positive_rate)}</div>
            <div><strong>Predicted wet rate:</strong> {formatMetric(metrics.predicted_positive_rate)}</div>
            <div><strong>Confusion:</strong> TN {tn ?? '-'} / FP {fp ?? '-'} / FN {fn ?? '-'} / TP {tp ?? '-'}</div>
            {url ? <div><strong>Metrics JSON:</strong> <a href={url} target="_blank" rel="noreferrer">open</a></div> : null}
          </div>
        </>
      ) : null}
    </div>
  );
}

function MaskPanel({ mask, error, url }: {
  mask?: Record<string, any> | null;
  error?: string | null;
  url?: string;
}) {
  if (!mask && !error) return null;

  return (
    <div className="evaluation-panel">
      <div className="result-card-title">Inference Mask</div>
      {error ? <div className="error-hint">{error}</div> : null}
      {mask ? (
        <>
          <div className="metric-grid">
            <div><strong>Valid pixels</strong><span>{formatMetric(mask.valid_pixels)}</span></div>
            <div><strong>Masked pixels</strong><span>{formatMetric(mask.masked_pixels)}</span></div>
            <div><strong>Valid rate</strong><span>{formatMetric(mask.valid_rate)}</span></div>
            <div><strong>Masked rate</strong><span>{formatMetric(mask.masked_rate)}</span></div>
          </div>
          <div className="metric-meta">
            <div><strong>Total pixels:</strong> {formatMetric(mask.total_pixels)}</div>
            {url ? <div><strong>Mask JSON:</strong> <a href={url} target="_blank" rel="noreferrer">open</a></div> : null}
          </div>
        </>
      ) : null}
    </div>
  );
}

export default function ResultsPage({ inference, onBackToRoi }: Props) {
  return (
    <section className="page-section">
      <div className="page-title-row">
        <h2>Results</h2>
        <button type="button" onClick={onBackToRoi}>
          Back to ROI Selection
        </button>
      </div>

      <div className="results-grid">
        <GroundTruthCard
          url={inference?.labelMapUrl}
          error={inference?.labelVisualizationError}
          fallbackUrl={inference?.pseudocolorUrl}
        />
        <WaterDetectionCard url={inference?.probabilityOverlayUrl} />
        <ResultCard title="Spatial Attention" url={inference?.spatialAttentionOverlayUrl} />
        <ResultCard title="Spectral Attention" url={inference?.spectralAttentionUrl} />
      </div>

      <MaskPanel
        mask={inference?.inferenceMask}
        error={inference?.inferenceMaskError}
        url={inference?.inferenceMaskUrl}
      />

      <EvaluationPanel
        metrics={inference?.evaluationMetrics}
        error={inference?.evaluationError}
        url={inference?.evaluationMetricsUrl}
      />
    </section>
  );
}
