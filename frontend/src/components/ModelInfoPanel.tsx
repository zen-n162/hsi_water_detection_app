import type { PreviewResponse, InferenceResponse } from '../lib/api';

export default function ModelInfoPanel({
  preview,
  inference,
}: {
  preview: PreviewResponse | null;
  inference: InferenceResponse | null;
}) {
  return (
    <details className="collapse" open>
      <summary>Resolved inference settings</summary>
      <div className="model-info">
        <div><strong>Run:</strong> {inference?.resolvedRunName || '-'}</div>
        <div><strong>Checkpoint:</strong> {inference?.resolvedModelCheckpoint || '-'}</div>
        <div><strong>Calibration:</strong> {inference?.resolvedTemperatureJson || '-'}</div>
        <div><strong>Threshold:</strong> {inference?.resolvedThreshold ?? '-'}</div>
        <div><strong>Manifest:</strong> {inference?.resolvedManifest || '-'}</div>
        <div><strong>Dataset:</strong> {inference?.resolvedDataset || '-'}</div>
        <div><strong>Device:</strong> {inference?.executedDevice || '-'}</div>
        <div><strong>Preview:</strong> {preview?.grayscalePreviewUrl ? 'loaded' : 'not loaded'}</div>
      </div>
    </details>
  );
}
